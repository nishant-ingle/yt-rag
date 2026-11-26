"""Incremental playlist sync and time-window chunker."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, AsyncGenerator

import yt_dlp
from openai import AsyncOpenAI
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from youtube_transcript_api._errors import IpBlocked, RequestBlocked

import backend.db as db
from backend.config import settings

logger = logging.getLogger(__name__)

ytt = YouTubeTranscriptApi()

_rate_limit_info: dict[str, Any] = {
    "is_rate_limited": False,
    "message": None,
    "timestamp": None,
}


def get_rate_limit_info() -> dict[str, Any]:
    """Return current rate limit status."""
    return dict(_rate_limit_info)


def set_rate_limited(message: str = "YouTube rate-limited transcript requests from this IP (429). Pausing sync to protect IP.") -> None:
    """Set rate limit flag, message and timestamp."""
    logger.warning("Rate limit flag set: %s", message)
    _rate_limit_info["is_rate_limited"] = True
    _rate_limit_info["message"] = message
    _rate_limit_info["timestamp"] = datetime.now(timezone.utc).isoformat()


def clear_rate_limited() -> None:
    """Reset rate limit status."""
    logger.info("Rate limit flag cleared.")
    _rate_limit_info["is_rate_limited"] = False
    _rate_limit_info["message"] = None
    _rate_limit_info["timestamp"] = None


def get_playlist_video_ids(playlist_url: str) -> list[dict[str, str]]:
    """Fetch video IDs and titles from a YouTube playlist URL using yt-dlp."""
    logger.info("Fetching video list from playlist: %s", playlist_url)
    opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower() or "blocked" in err_msg.lower():
                logger.warning("Rate limit hit while fetching playlist videos from %s: %s", playlist_url, err_msg)
                set_rate_limited("YouTube rate limit exceeded (HTTP 429) while fetching playlist videos.")
            else:
                logger.error("Failed to fetch playlist videos from %s: %s", playlist_url, err_msg)
            raise

    entries = info.get("entries") or []
    video_list = [
        {"video_id": e["id"], "title": e.get("title", "")}
        for e in entries
        if e and e.get("id")
    ]
    logger.info("Fetched %d videos from playlist %s.", len(video_list), playlist_url)
    return video_list


def check_playlist_updates(playlist_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Check configured playlists for new unindexed videos."""
    results: list[dict[str, Any]] = []

    for pl in settings.PLAYLISTS:
        if playlist_ids and pl["id"] not in playlist_ids:
            continue

        url = pl.get("url", "")
        pl_id = pl["id"]
        pl_name = pl.get("name", pl_id)

        try:
            entries = get_playlist_video_ids(url)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower() or "blocked" in err_msg.lower():
                logger.warning("Rate limit while checking updates for playlist '%s': %s", pl_name, err_msg)
                set_rate_limited("YouTube rate limit exceeded (HTTP 429) while checking playlist updates.")
            else:
                logger.error("Error fetching playlist '%s' (%s): %s", pl_name, url, err_msg)
            entries = []

        known_ids_in_playlist = db.get_known_video_ids(playlist_id=pl_id)
        new_videos = [e for e in entries if e["video_id"] not in known_ids_in_playlist]

        logger.info(
            "Playlist '%s': %d known, %d new video(s) pending indexing.",
            pl_name, len(known_ids_in_playlist), len(new_videos),
        )

        results.append({
            "id": pl_id,
            "name": pl_name,
            "url": url,
            "indexed_count": len(known_ids_in_playlist),
            "new_count": len(new_videos),
            "new_videos": new_videos,
        })

    return results


def fetch_video_details_and_transcript(video_id: str) -> dict[str, Any] | None:
    """Fetch video metadata via yt-dlp and transcript segments via youtube-transcript-api."""
    logger.info("Fetching metadata and transcript for video: %s", video_id)
    opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True
    }
    rate_limited = False
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            meta = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
        except Exception as e:
            meta = {}
            err_msg = str(e)
            if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower() or "blocked" in err_msg.lower():
                logger.warning("Rate limit hit while fetching metadata for video %s: %s", video_id, err_msg)
                rate_limited = True
            else:
                logger.warning("Could not fetch metadata for video %s: %s", video_id, err_msg)

    title = meta.get("title", "")
    upload_date = meta.get("upload_date", "")
    duration = int(meta.get("duration", 0) or 0)

    if title:
        logger.debug("Video %s metadata: title=%r, upload_date=%s, duration=%ds", video_id, title, upload_date, duration)

    segments = None
    is_missing_transcript = False

    try:
        # Prefers English (including Indian, US, GB variants)
        fetched = ytt.fetch(video_id, languages=["en", "en-IN", "en-US", "en-GB"])
        segments = fetched.to_raw_data()
        logger.debug("Fetched %d transcript segments for video %s.", len(segments), video_id)
    except (IpBlocked, RequestBlocked):
        logger.warning("IP/request blocked while fetching transcript for video %s.", video_id)
        rate_limited = True
    except (NoTranscriptFound, TranscriptsDisabled):
        logger.info("No English transcript available for video %s.", video_id)
        is_missing_transcript = True
    except Exception as e:
        err_msg = str(e)
        if "IpBlocked" in err_msg or "429" in err_msg or "Too Many Requests" in err_msg:
            logger.warning("Rate limit/block while fetching transcript for video %s: %s", video_id, err_msg)
            rate_limited = True
        else:
            logger.warning("Transcript fetch failed for video %s, trying fallback: %s", video_id, err_msg)
            try:
                # Fallback to finding transcript list
                tl = ytt.list(video_id)
                transcript = tl.find_transcript(["en", "en-IN", "en-US", "en-GB"])
                segments = transcript.fetch().to_raw_data()
                logger.debug("Fallback transcript fetch succeeded for video %s (%d segments).", video_id, len(segments))
            except (IpBlocked, RequestBlocked):
                logger.warning("IP/request blocked in fallback transcript fetch for video %s.", video_id)
                rate_limited = True
            except (NoTranscriptFound, TranscriptsDisabled):
                logger.info("No transcript found via fallback for video %s.", video_id)
                is_missing_transcript = True
            except Exception as e2:
                if "IpBlocked" in str(e2) or "429" in str(e2):
                    logger.warning("Rate limit in fallback transcript fetch for video %s: %s", video_id, e2)
                    rate_limited = True
                else:
                    logger.warning("Fallback transcript fetch failed for video %s: %s", video_id, e2)
                    is_missing_transcript = True

    return {
        "video_id": video_id,
        "title": title,
        "upload_date": upload_date,
        "duration": duration,
        "segments": segments,
        "rate_limited": rate_limited,
        "is_missing_transcript": is_missing_transcript,
    }


def chunk_segments(
    segments: list[dict[str, Any]],
    target_seconds: int = 90,
    overlap_seconds: int = 15,
) -> list[dict[str, Any]]:
    """Time-bounded sliding-window chunker preserving accurate timestamp bounds."""
    if not segments:
        return []

    chunks: list[dict[str, Any]] = []
    n = len(segments)
    i = 0

    while i < n:
        chunk_start = float(segments[i]["start"])
        first_seg = segments[i]
        curr_text: list[str] = []
        j = i

        while j < n:
            seg = segments[j]
            text = seg.get("text", "").strip()
            if text:
                curr_text.append(text)
            if (float(seg["start"]) - chunk_start) >= target_seconds:
                break
            j += 1

        last_seg = segments[min(j, n - 1)]
        start_sec = int(float(first_seg["start"]))
        end_sec = int(float(last_seg["start"]) + float(last_seg.get("duration", 0)))
        full_text = " ".join(curr_text).strip()

        if full_text:
            chunks.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "text": full_text,
            })

        if j >= n - 1:
            break

        step_target = max(chunk_start + 1.0, end_sec - overlap_seconds)
        next_i = i + 1
        while next_i < j and float(segments[next_i]["start"]) < step_target:
            next_i += 1
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks


async def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Batch embed text chunks using the configured LLM embedding model."""
    if not texts:
        return []

    embed_key = settings.EMBED_API_KEY or settings.LLM_API_KEY
    client = AsyncOpenAI(base_url=settings.EMBED_BASE_URL, api_key=embed_key)
    embeddings: list[list[float]] = []
    batch_size = 64
    num_batches = (len(texts) + batch_size - 1) // batch_size
    logger.debug("Embedding %d chunk(s) in %d batch(es) using model '%s'.", len(texts), num_batches, settings.EMBED_MODEL)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug("Embedding batch %d/%d (%d texts).", i // batch_size + 1, num_batches, len(batch))
        response = await client.embeddings.create(
            model=settings.EMBED_MODEL,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])

    logger.debug("Embedding complete: %d vector(s) returned.", len(embeddings))
    return embeddings


async def sync_playlists(
    playlist_ids: list[str] | None = None,
    max_videos_per_playlist: int | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Incremental playlist sync yielding real-time progress events."""
    logger.info("Starting sync: playlist_ids=%s, max_videos_per_playlist=%s", playlist_ids, max_videos_per_playlist)
    await asyncio.to_thread(db.init_db)
    updates = await asyncio.to_thread(check_playlist_updates, playlist_ids=playlist_ids)
    global_known_ids = await asyncio.to_thread(db.get_known_video_ids)

    # If already rate-limited and no pending videos could be found
    rl_info = get_rate_limit_info()
    total_pending = sum(len(p.get("new_videos", [])) for p in updates)
    if rl_info.get("is_rate_limited") and total_pending == 0:
        logger.warning("Sync aborted: already rate-limited and no new videos to process.")
        yield {
            "status": "rate_limited",
            "message": rl_info.get("message") or "YouTube rate-limited requests from this IP (429). Sync is paused to protect IP.",
        }
        return

    logger.info("Sync plan: %d playlist(s), %d total pending video(s).", len(updates), total_pending)

    for pl_info in updates:
        pl_id = pl_info["id"]
        pl_name = pl_info["name"]
        pending_videos = pl_info["new_videos"]

        if max_videos_per_playlist is not None:
            pending_videos = pending_videos[:max_videos_per_playlist]

        total_videos = len(pending_videos)
        logger.info("Processing playlist '%s': %d video(s) to index.", pl_name, total_videos)

        for idx, video_entry in enumerate(pending_videos, start=1):
            vid = video_entry["video_id"]
            video_title = video_entry.get("title", "")

            logger.info("[%d/%d] Processing video '%s' (%s) in playlist '%s'.", idx, total_videos, video_title, vid, pl_name)

            yield {
                "playlist": pl_name,
                "playlist_id": pl_id,
                "current": idx,
                "total": total_videos,
                "video_id": vid,
                "video_title": video_title,
                "status": "processing",
                "message": f"Processing video {idx}/{total_videos}",
            }

            # If video was already fetched in another playlist, simply link it
            if vid in global_known_ids:
                logger.info("Video %s already indexed globally, linking to playlist '%s'.", vid, pl_name)
                await asyncio.to_thread(db.link_playlist_video, playlist_id=pl_id, video_id=vid)
                yield {
                    "playlist": pl_name,
                    "playlist_id": pl_id,
                    "current": idx,
                    "total": total_videos,
                    "video_id": vid,
                    "video_title": video_title,
                    "status": "linked",
                    "message": "Already indexed in database, linked to playlist",
                }
                continue

            # Fetch details & transcript in worker thread
            details = await asyncio.to_thread(fetch_video_details_and_transcript, vid)
            if not details:
                logger.warning("No details returned for video %s, using fallback metadata.", vid)
                details = {
                    "video_id": vid,
                    "title": video_title,
                    "upload_date": "",
                    "duration": 0,
                    "segments": None,
                }

            if not details.get("title"):
                details["title"] = video_title

            if details.get("rate_limited"):
                rate_limit_msg = "YouTube rate-limited transcript requests from this IP (429). Pausing sync to protect IP."
                logger.warning("Rate limit detected during sync of video %s. Halting sync.", vid)
                set_rate_limited(rate_limit_msg)
                yield {
                    "playlist": pl_name,
                    "playlist_id": pl_id,
                    "current": idx,
                    "total": total_videos,
                    "video_id": vid,
                    "video_title": details.get("title", video_title),
                    "status": "rate_limited",
                    "message": rate_limit_msg,
                }
                # Do NOT mark video as no_transcript, stop loop to avoid mass failures
                return

            segments = details.get("segments")
            if not segments:
                if details.get("is_missing_transcript"):
                    # Video genuinely has no captions; record status='no_transcript' to avoid endlessly retrying
                    logger.info("Video '%s' (%s) has no transcript — marking as no_transcript.", details.get("title", video_title), vid)
                    await asyncio.to_thread(
                        db.upsert_video_and_chunks,
                        video_data={
                            "video_id": vid,
                            "title": details.get("title", video_title),
                            "upload_date": details.get("upload_date", ""),
                            "duration": details.get("duration", 0),
                            "raw_segments_json": "[]",
                            "status": "no_transcript",
                        },
                        chunks=[],
                        embeddings=[],
                        playlist_id=pl_id,
                    )
                    global_known_ids.add(vid)
                    yield {
                        "playlist": pl_name,
                        "playlist_id": pl_id,
                        "current": idx,
                        "total": total_videos,
                        "video_id": vid,
                        "video_title": details.get("title", video_title),
                        "status": "no_transcript",
                        "message": "No captions available, marked skipped",
                    }
                else:
                    logger.warning("Could not retrieve transcript for video %s — skipping (not marked).", vid)
                    yield {
                        "playlist": pl_name,
                        "playlist_id": pl_id,
                        "current": idx,
                        "total": total_videos,
                        "video_id": vid,
                        "video_title": details.get("title", video_title),
                        "status": "skipped",
                        "message": "Could not retrieve transcript, skipped for now without marking",
                    }
                continue

            # Chunk segments preserving timestamps
            chunks = chunk_segments(segments)
            chunk_texts = [c["text"] for c in chunks]
            logger.debug("Video %s chunked into %d segment(s).", vid, len(chunks))

            # Generate vector embeddings
            logger.debug("Embedding %d chunk(s) for video %s...", len(chunks), vid)
            embeddings = await embed_chunks(chunk_texts)

            # Atomically commit video, chunks, and embeddings
            await asyncio.to_thread(
                db.upsert_video_and_chunks,
                video_data={
                    "video_id": vid,
                    "title": details.get("title", video_title),
                    "upload_date": details.get("upload_date", ""),
                    "duration": details.get("duration", 0),
                    "raw_segments_json": segments,
                    "status": "indexed",
                },
                chunks=chunks,
                embeddings=embeddings,
                playlist_id=pl_id,
            )
            global_known_ids.add(vid)
            logger.info(
                "✓ Indexed video '%s' (%s): %d chunks saved to DB.",
                details.get("title", video_title), vid, len(chunks),
            )

            yield {
                "playlist": pl_name,
                "playlist_id": pl_id,
                "current": idx,
                "total": total_videos,
                "video_id": vid,
                "video_title": details.get("title", video_title),
                "status": "indexed",
                "chunk_count": len(chunks),
                "message": f"Successfully indexed with {len(chunks)} chunks",
            }

            # Polite delay between videos to prevent YouTube IP rate limiting (2.5s pacing)
            logger.debug("Waiting 2.5s before next video to respect YouTube rate limits...")
            await asyncio.sleep(2.5)

    logger.info("Sync complete — clearing any stale rate limit flag.")
    # Successfully completed without rate limits
    clear_rate_limited()
