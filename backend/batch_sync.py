"""Automated background batch sync script for yt-rag.

Usage:
  python -m backend.batch_sync --batch 28 --interval-hours 4
  python -m backend.batch_sync --batch 28 --once
"""
import argparse
import asyncio
import datetime
import sys

import backend.ingest as ingest


async def run_batch(batch_size: int = 28) -> int:
    """Run a single batch sync and return count of indexed videos."""
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting batch sync (max {batch_size} videos per playlist)...")
    indexed_count = 0
    rate_limited = False

    async for event in ingest.sync_playlists(max_videos_per_playlist=batch_size):
        status = event.get("status")
        pl = event.get("playlist", "")
        title = event.get("video_title", "")
        cur = event.get("current", 0)
        tot = event.get("total", 0)

        if status == "processing":
            print(f"  [{cur}/{tot}] Processing: {title} ({pl})")
        elif status == "indexed":
            indexed_count += 1
            chunks = event.get("chunk_count", 0)
            print(f"  ✓ Indexed: {title} ({chunks} chunks)")
        elif status == "rate_limited":
            rate_limited = True
            msg = event.get("message", "")
            print(f"  ⚠️ Rate limited: {msg}")
            break
        elif status == "no_transcript":
            print(f"  - Skipped (no transcript): {title}")

    if rate_limited:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch paused due to rate limits. Will retry on next scheduled interval.")
    else:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch complete! Newly indexed: {indexed_count} videos.")

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

    print(f"Starting scheduled batch sync: {args.batch} videos every {args.interval_hours} hours. Press Ctrl+C to stop.")
    while True:
        try:
            await run_batch(args.batch)
        except Exception as e:
            print(f"Error during batch sync: {e}")

        sleep_seconds = int(args.interval_hours * 3600)
        next_run = datetime.datetime.now() + datetime.timedelta(seconds=sleep_seconds)
        print(f"Sleeping until next batch at {next_run.strftime('%Y-%m-%d %H:%M:%S')} ({args.interval_hours}h)...")
        await asyncio.sleep(sleep_seconds)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBatch sync stopped by user.")
        sys.exit(0)
