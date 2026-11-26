"""Resumable re-embedding of all chunks using open-source embedding model (nomic-embed-text).

If interrupted, running this script again will continue from where it left off,
only embedding chunks that are not already present in `vec_chunks`.
"""
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


async def reembed_all(batch_size: int = 128, reset: bool = False):
    db_path = Path(settings.DB_PATH)
    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        return

    logger.info("Opening database at %s", db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    dim = settings.EMBED_DIM

    # 1. Ensure table exists or reset if explicitly requested
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'")
    table_def = cursor.fetchone()

    # If reset is requested or table doesn't match expected dim, recreate it
    needs_create = table_def is None or reset
    if table_def and f"float[{dim}]" not in table_def[0]:
        logger.warning("Existing vec_chunks dimension mismatch: %s. Recreating table with float[%d]...", table_def[0], dim)
        needs_create = True

    if needs_create:
        logger.info("Recreating vec_chunks table with float[%d]...", dim)
        with conn:
            conn.execute("DROP TABLE IF EXISTS vec_chunks")
            conn.execute(f"""
                CREATE VIRTUAL TABLE vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[{dim}]
                );
            """)

    # 2. Count total chunks and already embedded chunks
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vec_chunks")
    already_embedded = cursor.fetchone()[0]

    logger.info("Total chunks in db: %d | Already in vec_chunks: %d", total_chunks, already_embedded)

    # 3. Fetch ONLY chunks that have not yet been embedded
    cursor.execute("""
        SELECT chunk_id, text 
        FROM chunks 
        WHERE chunk_id NOT IN (SELECT chunk_id FROM vec_chunks)
        ORDER BY chunk_id
    """)
    rows = cursor.fetchall()
    remaining_chunks = len(rows)

    if remaining_chunks == 0:
        logger.info("All %d chunks are already embedded in vec_chunks! Nothing to do.", total_chunks)
        conn.close()
        return

    logger.info("Resuming embedding for remaining %d chunks...", remaining_chunks)

    # 4. Batch embed and insert
    api_key = settings.EMBED_API_KEY or settings.LLM_API_KEY or "ollama"
    client = AsyncOpenAI(base_url=settings.EMBED_BASE_URL, api_key=api_key)
    logger.info("Connecting to embedding model '%s' at %s", settings.EMBED_MODEL, settings.EMBED_BASE_URL)

    t0 = time.perf_counter()
    newly_embedded = 0

    for i in range(0, remaining_chunks, batch_size):
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

        newly_embedded += len(batch)
        current_total = already_embedded + newly_embedded
        elapsed = time.perf_counter() - t0
        rate = newly_embedded / elapsed if elapsed > 0 else 0
        pct = (current_total / total_chunks) * 100
        logger.info(
            "Progress: %d/%d (%.1f%% overall, +%d newly embedded). Rate: %.1f chunks/s. Elapsed: %.1fs",
            current_total, total_chunks, pct, newly_embedded, rate, elapsed,
        )

    conn.close()
    logger.info("Successfully finished! All %d chunks are now embedded in %.2fs.", total_chunks, time.perf_counter() - t0)


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    asyncio.run(reembed_all(reset=reset_flag))
