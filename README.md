# yt-rag: YouTube Playlist RAG & Opinion Synthesis Engine

> **Looking for a fully local setup?** If you don't want to use OpenAI APIs and have a local Llama setup, check out the [`local_llama`](https://github.com/nishant-ingle/yt-rag/tree/local_llama) branch.

A high-performance, embedded Retrieval-Augmented Generation (RAG) system designed to index YouTube playlists, extract timestamped transcripts, and answer nuanced questions with exact timestamp deep links, overall opinion synthesis, and longitudinal opinion evolution over time.

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

### Configuration
Configure your API keys and models in `.env`:
```env
# Cloud OpenAI configuration (default)
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-openai-api-key
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
```
*(Note: End users can also optionally provide their own OPENAI API key per session for chat queries via the BYOK card in the web UI sidebar).*

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

## Running with Docker

You can run `yt-rag` as a standalone container with zero local Python or Node.js dependencies installed—either by pulling the pre-built image from [Docker Hub](https://hub.docker.com/r/nishantingle/yt-rag) or building it locally.

### Option A: Pull Pre-built Image (Recommended)

Run directly using the official image on [Docker Hub](https://hub.docker.com/r/nishantingle/yt-rag):

```bash
docker run -p 8000:8000 --env-file .env nishantingle/yt-rag:latest

```

*(Or pass the API key directly without a `.env` file: `docker run -p 8000:8000 -e LLM_API_KEY=sk-... nishantingle/yt-rag:latest`)*

### Option B: Build from Source

If you want to build the image locally from your own changes:

```bash
docker build -t yt-rag .
docker run -p 8000:8000 --env-file .env yt-rag

```

Open [http://localhost:8000](http://localhost:8000) in your browser.

> [!TIP]
> **Persistent Data & Read-Write Mode**:
> By default, the container includes the pre-indexed database (`data/rag.db`) in read-only snapshot mode (`READ_ONLY_MODE=true`). To persist newly synced playlists or enable full write access, mount the `data` directory and set `READ_ONLY_MODE=false` (replace `nishantingle/yt-rag:latest` with `yt-rag` if built locally):
> ```bash
> docker run -p 8000:8000 \
>   -v $(pwd)/data:/app/data \
>   -e READ_ONLY_MODE=false \
>   --env-file .env \
>   nishantingle/yt-rag:latest
> 
> ```
> 
> 

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

3. **Multi-Provider BYOK & Local LLMs**:
   - Built-in multi-provider support: OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-3.8-flash`), Anthropic Claude (`claude-sonnet-4-6`), or local Ollama.
   - BYOK (Bring Your Own Key) sidebar configuration stored securely in session storage.

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
├── .env.example             # Template for OpenAI and Ollama configurations
├── Dockerfile               # Production multi-stage container
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
