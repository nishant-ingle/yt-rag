# yt-rag: YouTube Playlist RAG & Opinion Synthesis Engine

A high-performance, embedded Retrieval-Augmented Generation (RAG) system designed to index YouTube playlists, extract timestamped transcripts, and answer nuanced questions with exact timestamp deep links, overall opinion synthesis, and longitudinal opinion evolution over time.

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

3. **Zero-Friction LLM Switching**:
   - Supports both cloud OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) and local Ollama (`llama3.1:8b`, `nomic-embed-text`) through standard OpenAI-compatible endpoints with zero code modifications.

4. **Three Query Modes**:
   - **Timestamp Lookup**: Finds specific quotes and facts with clickable `https://youtu.be/{video_id}?t={start_sec}` links.
   - **Overall Opinion Synthesis**: Synthesizes the creator's view across distinct videos (capped at 2 chunks per video).
   - **Opinion Evolution Over Time**: Sorts matching chunks chronologically by `upload_date` to contrast early versus recent stances.

---

## Repository Layout

```text
yt-rag/
├── .env.example             # Template for OpenAI and Ollama configurations
├── .gitignore               # Ignored environments, cache, and db files
├── README.md                # Project documentation and guide
├── SPEC.md                  # Complete architectural and implementation specification
├── start.sh                 # One-command setup & launch script
├── backend/
│   ├── __init__.py
│   ├── config.py            # Pydantic settings & hardcoded playlists
│   ├── db.py                # SQLite + sqlite-vec + FTS5 storage layer & hybrid search
│   ├── ingest.py            # Incremental playlist sync & time-window chunker (Step 3)
│   ├── rag.py               # Query router & multi-mode RAG engine (Step 4)
│   ├── main.py              # FastAPI app (Step 5)
│   └── requirements.txt     # Pinned Python dependencies
├── data/
│   └── rag.db               # SQLite database (auto-created on db.init_db())
└── frontend/
    ├── package.json
    └── src/
        ├── App.tsx          # React UI (Step 6)
        └── api.ts
```

---

## Database Schema (Step 2)

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

### 2. Setup Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Configure `.env`
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

### 4. Initialize Database
```bash
python -c "import backend.db as db; db.init_db()"
```

### 5. Run Verification Checks
To verify the Step 1 & Step 2 implementations:
```bash
python -c "
import backend.db as db
from backend.config import settings
db.init_db()
print('Database initialized at:', settings.DB_PATH)
print('Configured playlists:', [p['name'] for p in settings.PLAYLISTS])
"
```

---

## Roadmap

- [x] **Step 1**: Project Scaffolding & Unified Configuration
- [x] **Step 2**: SQLite + `sqlite-vec` + FTS5 Storage Layer & Hybrid Search
- [x] **Step 3**: Incremental Ingestion & Timestamp-Preserving Chunker (`yt-dlp` + `youtube-transcript-api`)
- [x] **Step 4**: Multi-Mode RAG Engine (Lookup, Opinion, Evolution)
- [x] **Step 5**: FastAPI Backend & Single-Port Static Serving
- [x] **Step 6**: React + TypeScript + Tailwind UI & One-Command Launcher
