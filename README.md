# yt-rag: YouTube Playlist RAG & Opinion Synthesis Engine

A high-performance, embedded Retrieval-Augmented Generation (RAG) system designed to index YouTube playlists, extract timestamped transcripts, and answer nuanced questions with exact timestamp deep links, overall opinion synthesis, and longitudinal opinion evolution over time.

![yt-rag Opinion Synthesis UI](<images/opinion on string theory.png>)

Focused initially on **Abhijit Chavda**'s channel ([@AbhijitChavda](https://www.youtube.com/@AbhijitChavda)) with playlists:
- **#AskAbhijit**: `https://youtube.com/playlist?list=PLSw9nXpYo3m6Bb_MVBgd0jRE3U4hJvZkd`
- **Abhijit Chavda Podcast**: `https://youtube.com/playlist?list=PLSw9nXpYo3m7CUa1yvPZmxYCLkchGa9-e`

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
   - Built-in multi-provider support: OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-2.5-flash`), Anthropic Claude (`claude-sonnet-4-6`), or local Ollama.
   - BYOK (Bring Your Own Key) sidebar configuration stored securely in session storage.

4. **Multi-Mode RAG & Real-Time Streaming**:
   - **Timestamp Lookup**: Finds specific quotes and facts with clickable `https://youtu.be/{video_id}?t={start_sec}` deep links.
   - **Overall Opinion Synthesis**: Synthesizes the creator's view across distinct videos (capped at 2 chunks per video).
   - **Opinion Evolution Over Time**: Sorts matching chunks chronologically by `upload_date` to contrast early versus recent stances.
   - **Token Streaming (SSE)**: Streams tokens in real time (< 500ms TTFT) with live citations.

5. **Rate-Limit Resilience & Background Batch Sync**:
   - Polite request spacing (2.5s pacing) and automatic handling of YouTube HTTP 429 rate limits.
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
├── tests/
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
- `upsert_video_and_chunks(video_data, chunks, embeddings, playlist_id)`: Atomically commits video metadata, chunks, and vector embeddings in a single SQLite transaction.
- `get_playlist_stats()`: Retrieves counts of indexed, missing transcript, and total videos per playlist.
- `hybrid_search(query_embedding, query_text, playlist_id, date_from, date_to, limit)`: Runs vector search and FTS5 BM25 search in parallel, fusing results with Reciprocal Rank Fusion (RRF).

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Node.js 20+

### 2. One-Command Launch (Recommended)
Run the automated launch script, which configures `.env`, sets up the Python virtual environment, installs backend dependencies, builds the frontend UI, and launches the server:
```bash
./start.sh
```
Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

### Manual Setup (Alternative)

#### 1. Setup Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

#### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to configure your API key or Ollama endpoint:
```env
# Cloud OpenAI configuration (default)
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-openai-api-key
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536

# Local Ollama alternative:
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_API_KEY=ollama
# CHAT_MODEL=llama3.1:8b
# EMBED_MODEL=nomic-embed-text
# EMBED_DIM=768
```

*(Note: You can also use BYOK in the web UI sidebar to provide OpenAI, Gemini, or Claude API keys per session without setting server keys).*

#### 3. Build Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

#### 4. Run Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Background Batch Sync

To sync larger playlists incrementally in the background with automatic pacing to avoid YouTube rate limits:
```bash
# Sync a single batch of up to 28 videos per playlist
python -m backend.batch_sync --batch 28 --once

# Or run continuously on a recurring schedule (e.g., every 4 hours)
python -m backend.batch_sync --batch 28 --interval-hours 4
```

---

## Running Tests

Run the test suite with:
```bash
python -m unittest discover tests
```
