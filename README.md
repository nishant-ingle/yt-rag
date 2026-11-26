# yt-rag: Local YouTube Playlist RAG & Opinion Synthesis Engine

A high-performance, 100% local Retrieval-Augmented Generation (RAG) system powered by **Ollama** (`llama3.2:3b` + `nomic-embed-text`), designed to index YouTube playlists, extract timestamped transcripts, and answer nuanced questions with exact timestamp deep links, overall opinion synthesis, and longitudinal opinion evolution over time.

## yt-rag Opinion Synthesis UI
<video src="images/video.mp4" style="/*display: block; */margin: auto; width: 60rem; max-width: 100%; height: auto;" autoplay loop muted playsinline></video>

Focused initially on **Abhijit Chavda**'s channel ([@AbhijitChavda](https://www.youtube.com/@AbhijitChavda)) with playlists:
- **#AskAbhijit Show**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLSw9nXpYo3m6Bb_MVBgd0jRE3U4hJvZkd)
- **Abhijit Chavda Podcast**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLSw9nXpYo3m7CUa1yvPZmxYCLkchGa9-e)
- **Geopolitical Videos**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLSw9nXpYo3m5B0oy2aMBlmWZGvb35Vv7T)
- **Geostrategy with Abhijit Chavda**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLSw9nXpYo3m5gO6wo-CwCFQe7hos3MC_B)
- **The #IndianInterest Podcast**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLSw9nXpYo3m6nR38Csp5Ea61pnLdRxdn-)
- **Understanding America w/ Abhijit Chavda**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLSw9nXpYo3m432nSt6uOp9rWYlwCtcBVq)
- **Collaborations**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLSw9nXpYo3m7AJ1XdjXwAxsuWUf7hL3dR)

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- [Ollama](https://ollama.com/) installed and running locally

### Pull Required Models in Ollama
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Configuration
Copy `.env.example` to `.env` (pre-configured for local Ollama):
```env
# Local Ollama configuration (Default for local_llama branch)
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
CHAT_MODEL=llama3.2:3b

EMBED_BASE_URL=http://localhost:11434/v1
EMBED_API_KEY=ollama
EMBED_MODEL=nomic-embed-text
EMBED_DIM=768
```

### (Optional) Re-embed Existing Chunks
If `data/rag.db` currently contains embeddings from a different dimension model (e.g., 1536-dim), re-embed all chunks with `nomic-embed-text` (768-dim):
```bash
.venv/bin/python scripts/reembed.py
```

### Launch
```bash
./start.sh
```
That's it! The script automatically:
- Creates `.env` from `.env.example` (if missing)
- Sets up the Python virtual environment and installs dependencies
- Builds the React frontend UI
- Starts the server at [http://localhost:8000](http://localhost:8000)

---

## Key Architectural Decisions

1. **SQLite + `sqlite-vec` + FTS5**:
   - Single embedded database file (`data/rag.db`).
   - Zero background daemons required.
   - Native hybrid search combining BM25 keyword matching (`chunks_fts`) and vector similarity (`vec_chunks`) using **Reciprocal Rank Fusion (RRF)**.
   - Synchronized via SQL triggers for automatic FTS and vector index consistency.

2. **Zero-Loss Raw JSON Transcripts**:
   - Transcripts are fetched directly into memory via `youtube-transcript-api` and archived as raw JSON in SQLite.
   - No temporary `.srt` files on disk.
   - Allows instant offline re-indexing or re-chunking without hitting YouTube API rate limits.

3. **100% Local Inference via Ollama**:
   - Chat and synthesis: `llama3.2:3b` via Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`).
   - Semantic embeddings: `nomic-embed-text` (768-dim) generating embeddings locally.
   - No external API keys or cloud tokens required.

4. **Multi-Mode RAG & Real-Time Streaming**:
   - **Abhijit AI Mode**: 1st-person POV synthesis reflecting Abhijit Chavda's voice, convictions, and speaking style.
   - **Timestamp Lookup**: Finds specific quotes and facts with clickable `https://youtu.be/{video_id}?t={start_sec}` deep links.
   - **Overall Opinion Synthesis**: Synthesizes the creator's view across distinct videos (capped at 2 chunks per video).
   - **Opinion Evolution Over Time**: Sorts matching chunks chronologically by `upload_date` to contrast early versus recent stances.
   - **Freeform Mode**: Direct knowledge-based answers without persona prompting.
   - **Token Streaming (SSE)**: Streams tokens in real time (< 500ms TTFT) with live citations.

5. **Rate-Limit Resilience & Background Batch Sync**:
   - Polite request spacing and automatic handling of YouTube HTTP 429 rate limits.
   - Videos with unavailable transcripts are flagged (`no_transcript`) to prevent repeated sync failures.
   - Automated scheduled batch sync script (`backend/batch_sync.py`) for syncing large playlists over time.

---

## Repository Layout

```text
yt-rag/
├── images/
│   └── opinion on string theory.png # Screenshot demo
├── backend/
│   ├── __init__.py
│   ├── batch_sync.py        # Automated batch sync runner
│   ├── config.py            # Pydantic settings & hardcoded playlists
│   ├── db.py                # SQLite + sqlite-vec + FTS5 storage layer & hybrid search
│   ├── ingest.py            # Incremental playlist sync & time-window chunker
│   ├── rag.py               # Query router, multi-mode RAG engine & streaming
│   ├── main.py              # FastAPI app & static file server
│   └── requirements.txt     # Pinned Python dependencies
├── data/
│   └── rag.db               # SQLite database (auto-created on db.init_db())
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.tsx          # React UI with streaming & citations drawer
│       └── api.ts           # Frontend API client
├── scripts/
│   └── reembed.py           # Utility to re-embed all chunks with a new model
├── tests/
│   ├── test_abhijit_ai_mode.py
│   ├── test_freeform_mode.py
│   └── test_sync_rate_limit.py
├── .env.example             # Template for local Ollama configuration
├── render.yaml              # Render deployment configuration
├── start.sh                 # One-command setup & launch script
├── README.md                # Project documentation and guide
└── SPEC.md                  # Complete architectural and implementation specification
```

---

## Database Schema

- **`playlist_videos`**: `(playlist_id, video_id)` composite primary key mapping videos to playlists. Videos shared across playlists are indexed only once.
- **`videos`**: `video_id`, `title`, `upload_date`, `duration`, `raw_segments_json`, `status` (`'indexed'` or `'no_transcript'`), `synced_at`.
- **`chunks`**: `chunk_id`, `video_id`, `start_sec`, `end_sec`, `text`.
- **`chunks_fts`**: FTS5 virtual table indexing `chunks(text)` with automatic `INSERT`, `DELETE`, and `UPDATE` triggers.
- **`vec_chunks`**: `vec0` virtual table dynamically sized from `settings.EMBED_DIM` with automatic chunk deletion triggers.

### Core Database Methods (`backend/db.py`)
- `init_db()`: Initializes tables, virtual tables, indexes, and triggers.
- `get_known_video_ids(playlist_id=None)`: Fetches set of existing video IDs to enable fast incremental sync.
- `link_playlist_video(playlist_id, video_id)`: Associates an existing video with a new playlist without re-downloading or re-embedding.
- `upsert_video_and_chunks(video_data, chunks, embeddings, playlist_id)`: Atomically commits video metadata, chunks, and vector embeddings in a single SQLite transaction.
- `get_playlist_stats()`: Retrieves counts of indexed, missing transcript, and total videos per playlist.
- `hybrid_search(query_embedding, query_text, playlist_id, date_from, date_to, limit)`: Runs vector search and FTS5 BM25 search in parallel, fusing results with Reciprocal Rank Fusion (RRF).

---

## Background Batch Sync

To sync larger playlists incrementally in the background with automatic pacing to avoid YouTube rate limits:
```bash
# Sync a single batch of up to 28 videos per playlist
python -m backend.batch_sync --batch 28 --once

# Or run continuously on a recurring schedule (e.g., every 4 hours)
python -m backend.batch_sync --batch 28 --interval-hours 4
```

> [!NOTE]
> **Discrepancy Between Total and Indexed Video Counts**:
> You may notice that the total number of videos in a playlist (e.g., in sync logs) is higher than the `Indexed` count in the UI. This happens because some YouTube videos **do not have English transcripts or captions available** (e.g., captions disabled by the creator, livestreams, or non-English audio without auto-subtitles). `yt-rag` flags these as `status: 'no_transcript'` so they aren't repeatedly re-queried, while the UI only shows videos that have searchable transcript chunks and vector embeddings.

---

## Handling YouTube Rate Limits (HTTP 429 Workaround)

YouTube monitors automated transcript requests (`/api/timedtext`) and temporarily rate-limits unauthenticated IP addresses after approximately ~28 videos per day.

### Mobile Hotspot Reset Workaround
Because the rate limit is enforced strictly against your **public IP address**, you can bypass the block immediately without waiting 24 hours:

1. Connect your computer to your phone's **Personal Hotspot**.
2. On your phone, toggle **Airplane Mode ON**, wait 5 seconds, and toggle it **OFF**.
3. Your mobile carrier immediately assigns a **brand-new public IP address**.
4. In the `yt-rag` web UI, click **"Clear Rate Limit"** (or restart the server) and resume syncing another batch!

---

## Running Tests

Run the test suite with:
```bash
source .venv/bin/activate
python -m unittest discover tests
```
