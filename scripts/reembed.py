"""Re-embed all chunks in database using open-source embedding model (nomic-embed-text)."""
import asyncio
import logging
import sqlite3
import sys
import time
from pathlib import Path

import sqlite_vec
from openai import AsyncOpenAI

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def reembed_all(batch_size: int = 128):
    db_path = Path(settings.DB_PATH)
    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        return

    logger.info("Opening database at %s", db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # 1. Fetch all chunks
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
    rows = cursor.fetchall()
    total_chunks = len(rows)
    logger.info("Total chunks to embed: %d", total_chunks)

    if total_chunks == 0:
        logger.info("No chunks to embed.")
        conn.close()
        return

    # 2. Re-create vec_chunks virtual table with new dimension
    dim = settings.EMBED_DIM
    logger.info("Recreating vec_chunks table with float[%d]...", dim)
    with conn:
        conn.execute("DROP TABLE IF EXISTS vec_chunks")
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{dim}]
            );
        """)

    # 3. Batch embed and insert
    api_key = settings.EMBED_API_KEY or settings.LLM_API_KEY or "ollama"
    client = AsyncOpenAI(base_url=settings.EMBED_BASE_URL, api_key=api_key)
    logger.info("Connecting to embedding model '%s' at %s", settings.EMBED_MODEL, settings.EMBED_BASE_URL)

    t0 = time.perf_counter()
    embedded_count = 0

    for i in range(0, total_chunks, batch_size):
        batch = rows[i : i + batch_size]
        batch_ids = [r[0] for r in batch]
        batch_texts = [r[1] for r in batch]

        # Call Ollama embedding endpoint
        resp = await client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=batch_texts,
        )

        # Insert embeddings into vec_chunks
        insert_records = []
        for chunk_id, item in zip(batch_ids, resp.data):
            emb_blob = sqlite_vec.serialize_float32(item.embedding)
            insert_records.append((chunk_id, emb_blob))

        with conn:
            conn.executemany("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)", insert_records)

        embedded_count += len(batch)
        elapsed = time.perf_counter() - t0
        rate = embedded_count / elapsed
        pct = (embedded_count / total_chunks) * 100
        logger.info(
            "Progress: %d/%d (%.1f%%) chunks embedded. Rate: %.1f chunks/s. Elapsed: %.1fs",
            embedded_count, total_chunks, pct, rate, elapsed,
        )

    conn.close()
    logger.info("Successfully re-embedded all %d chunks in %.2fs!", embedded_count, time.perf_counter() - t0)


if __name__ == "__main__":
    asyncio.run(reembed_all())
