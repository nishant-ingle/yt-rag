"""Automated background batch sync script for yt-rag.

Usage:
  python -m backend.batch_sync --batch 28 --interval-hours 4
  python -m backend.batch_sync --batch 28 --once
"""
import argparse
import asyncio
import datetime
import logging
import sys

import backend.ingest as ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_batch(batch_size: int = 28) -> int:
    """Run a single batch sync and return count of indexed videos."""
    logger.info("Starting batch sync (max %d videos per playlist)...", batch_size)
    indexed_count = 0
    rate_limited = False

    async for event in ingest.sync_playlists(max_videos_per_playlist=batch_size):
        status = event.get("status")
        pl = event.get("playlist", "")
        title = event.get("video_title", "")
        cur = event.get("current", 0)
        tot = event.get("total", 0)

        if status == "processing":
            logger.info("[%d/%d] Processing: %s (%s)", cur, tot, title, pl)
        elif status == "indexed":
            indexed_count += 1
            chunks = event.get("chunk_count", 0)
            logger.info("✓ Indexed: %s (%d chunks)", title, chunks)
        elif status == "rate_limited":
            rate_limited = True
            msg = event.get("message", "")
            logger.warning("⚠️  Rate limited: %s", msg)
            break
        elif status == "no_transcript":
            logger.info("— Skipped (no transcript): %s", title)
        elif status == "skipped":
            logger.info("— Skipped (no data): %s", title)
        elif status == "linked":
            logger.info("⇢ Linked (already indexed): %s", title)

    if rate_limited:
        logger.warning("Batch paused due to rate limits. Will retry on next scheduled interval.")
    else:
        logger.info("Batch complete! Newly indexed: %d video(s).", indexed_count)

    return indexed_count


async def main():
    parser = argparse.ArgumentParser(description="yt-rag automated batch sync")
    parser.add_argument("--batch", type=int, default=28, help="Batch size of videos per playlist (default: 28)")
    parser.add_argument("--interval-hours", type=float, default=4.0, help="Interval between batches in hours (default: 4.0)")
    parser.add_argument("--once", action="store_true", help="Run a single batch and exit")

    args = parser.parse_args()


    if args.once:
        await run_batch(args.batch)
        return

    logger.info("Starting scheduled batch sync: %d videos every %.1f hour(s). Press Ctrl+C to stop.", args.batch, args.interval_hours)
    while True:
        try:
            await run_batch(args.batch)
        except Exception as e:
            logger.error("Error during batch sync: %s", e, exc_info=True)

        sleep_seconds = int(args.interval_hours * 3600)
        next_run = datetime.datetime.now() + datetime.timedelta(seconds=sleep_seconds)
        logger.info("Sleeping until next batch at %s (%.1fh)...", next_run.strftime("%Y-%m-%d %H:%M:%S"), args.interval_hours)
        await asyncio.sleep(sleep_seconds)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Batch sync stopped by user.")
        sys.exit(0)
