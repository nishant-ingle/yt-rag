"""SQLite + sqlite-vec + FTS5 storage layer."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Sequence

import sqlite_vec

from backend.config import settings

logger = logging.getLogger(__name__)


def get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Connect to SQLite database and load sqlite-vec extension."""
    target_path = Path(db_path or settings.DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Opening database connection: %s", target_path)

    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def get_db_context(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a connected SQLite database."""
    conn = get_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Initialize database tables, virtual tables, and triggers."""
    logger.info("Initializing database schema at: %s", db_path or settings.DB_PATH)
    with get_db_context(db_path) as conn:
        with conn:
            # 1. Playlist-to-video mapping table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS playlist_videos (
                    playlist_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    PRIMARY KEY (playlist_id, video_id)
                );
            """)

            # 2. Videos table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    upload_date TEXT,
                    duration INTEGER,
                    raw_segments_json TEXT,
                    status TEXT,
                    synced_at TEXT
                );
            """)

            # 3. Chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    start_sec INTEGER NOT NULL,
                    end_sec INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks(video_id);")

            # 4. Chunks FTS5 table
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='chunk_id'
                );
            """)

            # Triggers to keep chunks_fts synchronized with chunks
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
                END;
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.chunk_id, old.text);
                END;
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.chunk_id, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
                END;
            """)

            # 5. Vector chunks table sized dynamically from settings.EMBED_DIM
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[{settings.EMBED_DIM}]
                );
            """)

            # Trigger to remove embedding when chunk is deleted
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_vec_ad AFTER DELETE ON chunks BEGIN
                    DELETE FROM vec_chunks WHERE chunk_id = old.chunk_id;
                END;
            """)


def get_known_video_ids(playlist_id: str | None = None, db_path: str | Path | None = None) -> set[str]:
    """Return set of known video IDs (optionally scoped to a specific playlist)."""
    with get_db_context(db_path) as conn:
        if playlist_id:
            cursor = conn.execute(
                "SELECT video_id FROM playlist_videos WHERE playlist_id = ?",
                (playlist_id,),
            )
        else:
            cursor = conn.execute("SELECT video_id FROM videos")
        return {row[0] for row in cursor.fetchall()}


def link_playlist_video(playlist_id: str, video_id: str, db_path: str | Path | None = None) -> None:
    """Link a video to a playlist if not already linked."""
    with get_db_context(db_path) as conn:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO playlist_videos (playlist_id, video_id) VALUES (?, ?)",
                (playlist_id, video_id),
            )


def upsert_video_and_chunks(
    video_data: dict[str, Any],
    chunks: list[dict[str, Any]] | None = None,
    embeddings: Sequence[Any] | None = None,
    playlist_id: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Atomically upsert video details, chunks, and embeddings."""
    video_id = video_data["video_id"]
    title = video_data.get("title", "")
    upload_date = video_data.get("upload_date", "")
    duration = int(video_data.get("duration", 0) or 0)
    status = video_data.get("status", "indexed")

    logger.debug(
        "Upserting video '%s' (%s): status=%s, chunks=%d, playlist=%s",
        title, video_id, status, len(chunks) if chunks else 0, playlist_id,
    )

    raw_segments = video_data.get("raw_segments_json", "[]")
    if not isinstance(raw_segments, str):
        raw_segments = json.dumps(raw_segments)

    synced_at = video_data.get("synced_at") or datetime.now(timezone.utc).isoformat()

    with get_db_context(db_path) as conn:
        with conn:
            # Upsert video metadata
            conn.execute(
                """
                INSERT INTO videos (video_id, title, upload_date, duration, raw_segments_json, status, synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    title = excluded.title,
                    upload_date = excluded.upload_date,
                    duration = excluded.duration,
                    raw_segments_json = excluded.raw_segments_json,
                    status = excluded.status,
                    synced_at = excluded.synced_at
                """,
                (video_id, title, upload_date, duration, raw_segments, status, synced_at),
            )

            # Link playlist if provided
            if playlist_id:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO playlist_videos (playlist_id, video_id)
                    VALUES (?, ?)
                    """,
                    (playlist_id, video_id),
                )

            # Remove existing chunks for this video (triggers clean up FTS and vec tables)
            conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))

            # Insert new chunks & embeddings
            if chunks:
                for idx, chunk in enumerate(chunks):
                    cursor = conn.execute(
                        """
                        INSERT INTO chunks (video_id, start_sec, end_sec, text)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            video_id,
                            int(chunk["start_sec"]),
                            int(chunk["end_sec"]),
                            chunk["text"],
                        ),
                    )
                    chunk_id = cursor.lastrowid

                    if embeddings is not None and idx < len(embeddings):
                        emb_serialized = sqlite_vec.serialize_float32(embeddings[idx])
                        conn.execute(
                            """
                            INSERT INTO vec_chunks (chunk_id, embedding)
                            VALUES (?, ?)
                            """,
                            (chunk_id, emb_serialized),
                        )


def get_playlist_stats(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return sync statistics for all configured and tracked playlists."""
    with get_db_context(db_path) as conn:
        cursor = conn.execute("""
            SELECT
                pv.playlist_id,
                COUNT(DISTINCT pv.video_id) as total_count,
                COUNT(DISTINCT CASE WHEN v.status = 'indexed' THEN v.video_id END) as indexed_count,
                COUNT(DISTINCT CASE WHEN v.status = 'no_transcript' THEN v.video_id END) as no_transcript_count
            FROM playlist_videos pv
            LEFT JOIN videos v ON pv.video_id = v.video_id
            GROUP BY pv.playlist_id
        """)
        stats_map = {row["playlist_id"]: dict(row) for row in cursor.fetchall()}

    results = []
    seen = set()

    for pl in settings.PLAYLISTS:
        pid = pl["id"]
        seen.add(pid)
        s = stats_map.get(pid, {})
        results.append({
            "id": pid,
            "name": pl.get("name", pid),
            "url": pl.get("url", ""),
            "total_count": s.get("total_count", 0),
            "indexed_count": s.get("indexed_count", 0),
            "no_transcript_count": s.get("no_transcript_count", 0),
        })

    for pid, s in stats_map.items():
        if pid not in seen:
            results.append({
                "id": pid,
                "name": pid,
                "url": "",
                "total_count": s.get("total_count", 0),
                "indexed_count": s.get("indexed_count", 0),
                "no_transcript_count": s.get("no_transcript_count", 0),
            })

    return results


def sanitize_fts_query(query: str) -> str:
    """Sanitize freeform query text into safe FTS5 boolean query."""
    tokens = re.findall(r"\w+", query)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def hybrid_search(
    query_embedding: Sequence[float] | None = None,
    query_text: str | None = None,
    playlist_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 12,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search combining sqlite-vec vector similarity and FTS5 BM25 with Reciprocal Rank Fusion."""
    logger.debug(
        "Hybrid search: query_text=%r, playlist_id=%s, date_from=%s, date_to=%s, limit=%d",
        (query_text or "")[:80], playlist_id, date_from, date_to, limit,
    )
    k_candidates = max(limit * 4, 60)
    vec_results: list[tuple[int, float]] = []
    fts_results: list[tuple[int, float]] = []

    with get_db_context(db_path) as conn:
        # 1. Vector Search
        if query_embedding is not None:
            try:
                emb_blob = sqlite_vec.serialize_float32(query_embedding)
                vec_cursor = conn.execute(
                    """
                    SELECT DISTINCT c.chunk_id, vc.distance
                    FROM vec_chunks vc
                    JOIN chunks c ON c.chunk_id = vc.chunk_id
                    JOIN videos v ON v.video_id = c.video_id
                    LEFT JOIN playlist_videos pv ON pv.video_id = v.video_id
                    WHERE vc.embedding MATCH ? AND k = ?
                      AND (? IS NULL OR pv.playlist_id = ?)
                      AND (? IS NULL OR v.upload_date >= ?)
                      AND (? IS NULL OR v.upload_date <= ?)
                    ORDER BY vc.distance ASC
                    LIMIT ?
                    """,
                    (
                        emb_blob,
                        k_candidates,
                        playlist_id,
                        playlist_id,
                        date_from,
                        date_from,
                        date_to,
                        date_to,
                        k_candidates,
                    ),
                )
                vec_results = [(row[0], float(row[1])) for row in vec_cursor.fetchall()]
            except Exception as e:
                logger.warning("Vector search failed (%s), relying on FTS5 BM25 search.", e)
                vec_results = []

        # 2. FTS5 BM25 Search
        clean_fts_query = sanitize_fts_query(query_text or "")
        if clean_fts_query:
            fts_cursor = conn.execute(
                """
                SELECT DISTINCT c.chunk_id, f.rank
                FROM chunks_fts f
                JOIN chunks c ON c.chunk_id = f.rowid
                JOIN videos v ON v.video_id = c.video_id
                LEFT JOIN playlist_videos pv ON pv.video_id = v.video_id
                WHERE chunks_fts MATCH ?
                  AND (? IS NULL OR pv.playlist_id = ?)
                  AND (? IS NULL OR v.upload_date >= ?)
                  AND (? IS NULL OR v.upload_date <= ?)
                ORDER BY f.rank ASC
                LIMIT ?
                """,
                (
                    clean_fts_query,
                    playlist_id,
                    playlist_id,
                    date_from,
                    date_from,
                    date_to,
                    date_to,
                    k_candidates,
                ),
            )
            fts_results = [(row[0], float(row[1])) for row in fts_cursor.fetchall()]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_k = 60.0
        rrf_scores: dict[int, float] = {}

        for rank, (chunk_id, _) in enumerate(vec_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (chunk_id, _) in enumerate(fts_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        if not rrf_scores:
            logger.debug("Hybrid search: no results from either vector or FTS search.")
            return []

        # Top chunk IDs by RRF score
        sorted_top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        top_ids = [item[0] for item in sorted_top]

        # 4. Fetch rich metadata for the top chunks
        placeholders = ",".join("?" for _ in top_ids)
        detail_cursor = conn.execute(
            f"""
            SELECT c.chunk_id, c.video_id, v.title, v.upload_date, c.start_sec, c.end_sec, c.text
            FROM chunks c
            JOIN videos v ON v.video_id = c.video_id
            WHERE c.chunk_id IN ({placeholders})
            """,
            top_ids,
        )
        chunk_map = {row["chunk_id"]: dict(row) for row in detail_cursor.fetchall()}

        # Reconstruct in order of fused ranking
        final_results = []
        for chunk_id, score in sorted_top:
            if chunk_id in chunk_map:
                item = dict(chunk_map[chunk_id])
                item["score"] = score
                final_results.append(item)

        logger.debug(
            "Hybrid search: vec=%d hits, fts=%d hits, rrf_candidates=%d, returned=%d results.",
            len(vec_results), len(fts_results), len(rrf_scores), len(final_results),
        )
        return final_results
