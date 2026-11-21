### Architecture Decisions & Key Trade-offs

#### 1. Handling Your Three RAG Query Modes

Standard top-$k$ vector search handles finding specific quotes, but fails on longitudinal synthesis ("how did his stance on X change from 2022 to 2026?"). Implement a **Query Router** in the backend that classifies the prompt into one of three retrieval strategies:

* **Timestamp Lookup (Fact/Quote Search):** Hybrid search (vector similarity + SQLite FTS5 keyword match for proper nouns like *S-400*, *Chola*, *BRICS*) returning top-8 chunks with direct deep links (`[https://youtu.be/](https://youtu.be/){video_id}?t={start_sec}`).
* **Overall Opinion Synthesis:** Retrieve top-15 diverse chunks across distinct videos (`GROUP BY video_id` cap of 2 chunks per video) so one 2-hour stream doesn't monopolize context.
* **Opinion Evolution Over Time:** Extract topic keywords, filter matching chunks sorted chronologically (`ORDER BY upload_date ASC`), bucket them by year/half-year, and run a map-reduce prompt comparing early vs. recent positions with cited dates and timestamps.

#### 2. Should You Delete `.srt` Files After Indexing?

**Do not write `.srt` files to disk at all—and do not discard the raw transcript.**

* A 1-hour transcript is ~10,000 words (~45 KB uncompressed, ~12 KB gzipped). Even 500 hours of podcasts is under **10 MB** inside SQLite.
* Fetch structured segments (`start`, `duration`, `text`) directly in memory via `youtube-transcript-api` and store them in a `videos` table.
* If you discard raw transcripts and later switch embedding models (e.g., `text-embedding-3-small` to local `nomic-embed-text`) or tweak chunk sizes, you would have to re-scrape YouTube—which triggers IP rate-limiting. Storing raw JSON in SQLite lets you re-index offline in seconds.

#### 3. SQLite Vector Store vs. Alternatives

| Option | Verdict | Why |
| --- | --- | --- |
| **SQLite + `sqlite-vec` + FTS5** | **Recommended** | Single `.db` file, zero background daemon, ACID-compliant sync tracking, native SQL filtering (`WHERE playlist_id = ? AND upload_date >= ?`), and built-in BM25 keyword search alongside vector search. |
| **LanceDB** | Runner-up | Great embedded columnar vector store with native hybrid search, though relational sync tables are cleaner in SQLite. |
| **ChromaDB / Qdrant** | Overkill | Adds heavier dependencies or separate server processes that break the "instant `git clone` and run" requirement. |

#### 4. Zero-Friction OpenAI / Local Llama Switching

Use the standard `openai` Python SDK for **both** cloud OpenAI and local Llama (via Ollama or LM Studio). Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`, meaning a single `.env` toggle switches both chat and embeddings with zero branching logic in code.

---

### Core Ingestion Logic (Playlist + Timestamped Transcripts)

`yt-dlp` with `extract_flat` fetches all video IDs in a playlist in ~1 second without downloading media. Your sync engine compares those IDs against SQLite, and **only** fetches full metadata (`upload_date`, `title`) and transcripts (`youtube-transcript-api` v1.2+ `fetch()` syntax) for new videos:

```python
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

ytt = YouTubeTranscriptApi()

def get_playlist_video_ids(playlist_url: str) -> list[dict]:
    opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    return [
        {"video_id": e["id"], "title": e.get("title", "")}
        for e in info.get("entries", []) if e and e.get("id")
    ]

def fetch_video_details_and_transcript(video_id: str) -> dict | None:
    opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        meta = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    try:
        # Prefers manual/auto English (including Indian English), falls back if needed
        fetched = ytt.fetch(video_id, languages=["en", "en-IN", "en-US", "en-GB"])
        segments = fetched.to_raw_data()  # [{"text": ..., "start": float, "duration": float}]
    except Exception:
        return None
    return {
        "video_id": video_id,
        "title": meta.get("title", ""),
        "upload_date": meta.get("upload_date", ""),  # YYYYMMDD string
        "duration": meta.get("duration", 0),
        "segments": segments,
    }

```

---

### Agent-Ready Build Specification

Forward the steps below directly to your coding agent.

#### Repository Layout Target

```text
yt-rag/
├── .env.example
├── start.sh                 # One-command setup & launch
├── backend/
│   ├── requirements.txt
│   ├── config.py            # Hardcoded playlists + env loader
│   ├── db.py                # SQLite + sqlite-vec + FTS5 schema & queries
│   ├── ingest.py            # Incremental playlist sync + time-window chunker
│   ├── rag.py               # Query router (lookup / opinion / timeline)
│   └── main.py              # FastAPI app (serves API + built React static files)
└── frontend/
    ├── package.json
    └── src/
        ├── App.tsx          # Split view: Playlist Sync Drawer + RAG Chat UI
        └── api.ts

```

---

#### Step 1: Project Scaffolding & Unified Configuration

* **Pre-conditions:** Python 3.11+ and Node.js 20+ installed locally.
* **Sub-tasks:**
1. Initialize the directory structure above and create `.env.example` supporting both OpenAI and Ollama:
```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here
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


2. Create `backend/config.py` using `pydantic-settings` to load `.env` and define `PLAYLISTS`:
```python
PLAYLISTS = [
    {"id": "geopolitics", "name": "Geopolitics", "url": "https://www.youtube.com/playlist?list=PL..."},
    {"id": "ama", "name": "Ask Me Anything", "url": "https://www.youtube.com/playlist?list=PL..."},
    {"id": "history", "name": "Indian History", "url": "https://www.youtube.com/playlist?list=PL..."},
]

```


3. Pin crisp dependencies in `backend/requirements.txt`: `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `yt-dlp`, `youtube-transcript-api>=1.2.0`, `openai`, `sqlite-vec`, `numpy`.


* **Acceptance Criteria:**
* Importing `backend.config.settings` loads `.env` cleanly and exposes `PLAYLISTS`, `LLM_BASE_URL`, `CHAT_MODEL`, `EMBED_MODEL`, and `EMBED_DIM`.
* Changing `LLM_BASE_URL` and `EMBED_DIM` in `.env` requires zero code changes.



---

#### Step 2: SQLite + `sqlite-vec` + FTS5 Storage Layer

* **Pre-conditions:** Step 1 completed; `sqlite-vec` installed.
* **Sub-tasks:**
1. In `backend/db.py`, initialize `data/rag.db` with `sqlite3`, loading the `sqlite_vec` extension on connection.
2. Create four tables:
* `playlist_videos(playlist_id TEXT, video_id TEXT, PRIMARY KEY (playlist_id, video_id))` — maps playlists to videos so shared videos across playlists are never downloaded or embedded twice.
* `videos(video_id TEXT PRIMARY KEY, title TEXT, upload_date TEXT, duration INT, raw_segments_json TEXT, status TEXT, synced_at TEXT)` — stores raw JSON segments (`status` = `'indexed'` or `'no_transcript'`).
* `chunks(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT, start_sec INT, end_sec INT, text TEXT)` + virtual table `chunks_fts USING fts5(text, content='chunks', content_rowid='chunk_id')` with insert/delete triggers.
* `vec_chunks USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[EMBED_DIM])` — virtual vector table sized dynamically from `settings.EMBED_DIM`.


3. Implement helper methods: `get_known_video_ids()`, `upsert_video_and_chunks()`, `get_playlist_stats()`, and `hybrid_search(query_embedding, query_text, playlist_id=None, date_from=None, date_to=None, limit=12)` using Reciprocal Rank Fusion (RRF) over `vec_chunks` cosine distance and `chunks_fts` BM25 rank.


* **Acceptance Criteria:**
* Running `python -c "import backend.db as db; db.init_db()"` creates `data/rag.db` with all tables and triggers without errors.
* Inserting a mock video with 2 chunks populates `videos`, `chunks`, `chunks_fts`, and `vec_chunks` atomically, and `hybrid_search` returns ranked results with `video_id`, `title`, `upload_date`, and `start_sec`.



---

#### Step 3: Incremental Ingestion & Timestamp-Preserving Chunker

* **Pre-conditions:** Step 2 completed.
* **Sub-tasks:**
1. In `backend/ingest.py`, implement `check_playlist_updates()` which runs `yt-dlp` (`extract_flat`) on every configured playlist in `PLAYLISTS`, compares against `videos` and `playlist_videos` in SQLite, and returns exact pending counts per playlist: `[{"id": ..., "name": ..., "indexed_count": N, "new_count": M, "new_videos": [...]}]`.
2. Implement a **time-bounded sliding-window chunker** (`chunk_segments(segments, target_seconds=90, overlap_seconds=15)`):
* Concatenate consecutive transcript segments until `segment.start - chunk_start >= 90` seconds (~220–260 words).
* Record exact `start_sec = int(first_seg["start"])` and `end_sec = int(last_seg["start"] + last_seg["duration"])`.
* Step back 15 seconds for the next chunk to prevent topic splits at boundaries.


3. Implement `sync_playlists(playlist_ids: list[str] | None = None)` as a generator/async stream yielding progress events (`{"playlist": ..., "current": i, "total": n, "video_title": ..., "status": ...}`):
* If `video_id` already exists in `videos`, only insert the `(playlist_id, video_id)` mapping.
* Otherwise, fetch metadata + transcript, chunk text, batch-embed chunks via `AsyncOpenAI(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY).embeddings.create(...)`, and commit to SQLite.
* If a video lacks captions, record `status='no_transcript'` so future syncs skip it instead of retrying endlessly.




* **Acceptance Criteria:**
* First sync downloads and indexes new videos; immediately running `check_playlist_updates()` afterward returns `new_count: 0` for all playlists.
* Every stored chunk has an accurate `start_sec` within ±2 seconds of the spoken sentence on YouTube.
* No `.srt` or temporary files remain on disk after sync.



---

#### Step 4: Multi-Mode RAG Engine (Search, Overall Opinion, Opinion Evolution)

* **Pre-conditions:** Step 3 completed; at least 2 videos indexed for testing.
* **Sub-tasks:**
1. In `backend/rag.py`, implement `classify_query(question: str) -> Literal["lookup", "opinion", "evolution"]` using a lightweight structured LLM call (or fast heuristic + LLM fallback).
2. Implement the three retrieval & synthesis pipelines:
* **`lookup`:** Run `hybrid_search(..., limit=8)`. Prompt LLM to answer concisely and cite exact timestamps inline as `[Video Title @ MM:SS](https://youtu.be/{video_id}?t={start_sec})`.
* **`opinion`:** Run `hybrid_search(..., limit=24)`, deduplicate to max 2 chunks per `video_id` (top 12 chunks total), and synthesize the speaker's consistent thesis, nuances, and supporting citations.
* **`evolution`:** Run `hybrid_search(..., limit=30)` with lower similarity threshold, sort retrieved chunks strictly ascending by `upload_date`, group into chronological periods (e.g., by year), and instruct the LLM to compare earlier statements vs. later statements, explicitly highlighting shifts, continuations, and dates.


3. Return structured JSON: `{"answer": str, "mode": str, "sources": [{"video_id", "title", "upload_date", "start_sec", "timestamp_str", "url", "snippet"}]}`.


* **Acceptance Criteria:**
* Asking *"What did he say about the BrahMos export to the Philippines?"* triggers `lookup` mode and returns clickable `youtu.be/...?t=...` links.
* Asking *"How has his view on US-India ties changed from 2022 to 2026?"* triggers `evolution` mode and orders evidence chronologically by `upload_date`.



---

#### Step 5: FastAPI Backend & Single-Port Static Serving

* **Pre-conditions:** Steps 3 and 4 completed.
* **Sub-tasks:**
1. In `backend/main.py`, expose clean REST/SSE endpoints:
* `GET /api/playlists` — returns hardcoded playlists, indexed video counts, and optionally checks YouTube for `new_count` (`?check_remote=true`).
* `POST /api/sync` — triggers incremental sync (streams Server-Sent Events progress so the UI shows live `Syncing 3/14: <Title>`).
* `POST /api/chat` — accepts `{"question": str, "playlist_id": str | None, "mode": "auto" | "lookup" | "opinion" | "evolution"}` and returns the RAG response + timestamped sources.


2. Mount `frontend/dist` at `/` via FastAPI `StaticFiles(html=True)` when built, so production/local users only run **one server** on `http://localhost:8000`.


* **Acceptance Criteria:**
* `uvicorn backend.main:app --port 8000` starts cleanly and serves OpenAPI docs at `/docs`.
* `POST /api/sync` streams progress events and updates SQLite counts.



---

#### Step 6: Minimal React Frontend & One-Command Launcher

* **Pre-conditions:** Step 5 completed.
* **Sub-tasks:**
1. Scaffold a minimal Vite + React + TypeScript + Tailwind CSS app in `frontend/` (keep it to 2–3 clean files, ~250 lines total).
2. Build a crisp two-panel interface:
* **Left Sidebar (Playlist Sync Manager):** Lists each hardcoded playlist with `[Indexed: X | New: Y]` badges, a `"Check for New"` button, and a `"Sync All New"` button displaying a live progress bar during ingestion.
* **Main Panel (RAG Workspace):** Playlist scope dropdown (`All Playlists` or specific playlist), Mode selector (`Auto`, `Find Timestamp`, `Overall Opinion`, `Opinion Over Time`), chat input, Markdown answer renderer with clickable timestamp badges, and collapsible source cards with embedded timestamp links.


3. Create `start.sh` in the repository root:
```bash
#!/usr/bin/env bash
set -e
[ ! -f .env ] && cp .env.example .env && echo "Created .env — add your OPENAI_API_KEY or Ollama config."
python3 -m venv .venv && source .venv/bin/activate
pip install -q -r backend/requirements.txt
if [ ! -d "frontend/dist" ]; then
  (cd frontend && npm install --silent && npm run build)
fi
uvicorn backend.main:app --host 0.0.0.0 --port 8000

```




* **Acceptance Criteria:**
* Fresh `git clone` -> edit `.env` -> `./start.sh` builds the frontend, starts FastAPI on port `8000`, and serves a working UI where clicking **"Check for New"** shows exact unindexed video counts per playlist and **"Sync"** ingests them into SQLite.


#### Step 7: BYOK Multi-Provider UI (OpenAI / Gemini / Claude), Read-Only Guard & Cloud Packaging (`yt-rag`)

* **Pre-conditions:** Steps 1–6 completed; `yt-rag` runs locally via `./start.sh` with multi-creator playlists configured and at least one playlist indexed into `data/rag.db` using `text-embedding-3-small` (`1536` dims).
* **Sub-tasks:**
1. **Add BYOK & Read-Only Configuration (`backend/config.py`, `.env.example`):**
* Keep embeddings fixed to `EMBED_BASE_URL: str = "[https://api.openai.com/v1](https://api.openai.com/v1)"`, `EMBED_MODEL: str = "text-embedding-3-small"`, and `EMBED_DIM: int = 1536` so `vec_chunks` dimensions remain invariant across deployments.
* Add `EMBED_API_KEY: str = ""` (falls back to `LLM_API_KEY` if unset), `READ_ONLY_MODE: bool = False`, and `ADMIN_TOKEN: str = ""`.
* Make `LLM_API_KEY: str = ""` optional so the server boots cleanly in public BYOK mode without a hardcoded chat key.


2. **Support Per-Request Provider Routing (`backend/rag.py`, `backend/main.py`):**
* Define a strict 3-provider allowlist in `backend/config.py` (never accept arbitrary user-supplied base URLs to prevent SSRF):
```python
CHAT_PROVIDERS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash"},
    "claude": {"base_url": "https://api.anthropic.com/v1/", "model": "claude-sonnet-4-6"},
}

```


* In `POST /api/chat`, read optional headers `X-Provider` (`"openai" | "gemini" | "claude"`, defaulting to `"openai"`) and `X-Api-Key`.
* Resolve the chat key as `request_api_key or settings.LLM_API_KEY` (return `HTTP 401` `"Enter an API key in the sidebar to ask questions"` if neither exists) and instantiate a per-request `AsyncOpenAI` client using the selected provider's `base_url` and `model`. Never log or persist `X-Api-Key`.
* Resolve the query embedding key as `settings.EMBED_API_KEY or settings.LLM_API_KEY or (request_api_key if provider == "openai" else "")` and embed queries against `settings.EMBED_BASE_URL` with `settings.EMBED_MODEL` (`1536` dims). Return `HTTP 401` if no OpenAI-capable embedding key is configured when a visitor uses Gemini or Claude.
* Protect `POST /api/sync` and `GET /api/playlists?check_remote=true`: when `READ_ONLY_MODE=true`, reject requests unless `X-Admin-Token` matches `settings.ADMIN_TOKEN` (`HTTP 403`). Expose `{"read_only_mode": settings.READ_ONLY_MODE, "has_server_key": bool(settings.LLM_API_KEY)}` via `GET /api/config`.


3. **Add Provider & API Key Controls to React UI (`frontend/src/`):**
* In the sidebar header, add an **LLM Provider** `<select>` with exactly 3 options (`OpenAI`, `Gemini`, `Claude`) and a password-masked **API Key** input whose placeholder dynamically updates (`"OPENAI API KEY"`, `"GEMINI API KEY"`, `"CLAUDE API KEY"`).
* Store `provider` and `apiKey` strictly in `sessionStorage` (cleared automatically when the browser tab closes) and pass them on `/api/chat` via `X-Provider` and `X-Api-Key` headers.
* When `GET /api/config` returns `read_only_mode: true`, disable/hide **"Check for New"** and **"Sync All New"** buttons and display a `"Hosted Snapshot (Read-Only)"` badge next to creator playlist counts.


4. **Create `.dockerignore`, Multi-Stage `Dockerfile`, and `render.yaml`:**
* `.dockerignore` (excludes local artifacts while **including `data/rag.db**` for ephemeral cloud containers):
```text
.venv
__pycache__
*.pyc
frontend/node_modules
frontend/dist
.env
.git

```


* `Dockerfile` (builds React in Node 20, copies `dist/`, `backend/`, and pre-synced `data/` into `python:3.11-slim`):
```dockerfile
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV PORT=8000
ENV READ_ONLY_MODE=true
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]

```


* `render.yaml` (Render Blueprint for one-click free Docker web service deployment):
```yaml
services:
  - type: web
    name: yt-rag
    runtime: docker
    plan: free
    dockerfilePath: ./Dockerfile
    envVars:
      - key: EMBED_BASE_URL
        value: https://api.openai.com/v1
      - key: EMBED_API_KEY
        sync: false
      - key: EMBED_MODEL
        value: text-embedding-3-small
      - key: EMBED_DIM
        value: "1536"
      - key: READ_ONLY_MODE
        value: "true"

```






* **Acceptance Criteria:**
* Selecting `OpenAI`, `Gemini`, or `Claude` in the UI and entering a valid key for that provider routes `POST /api/chat` to the corresponding OpenAI-compatible endpoint and returns cited YouTube timestamp answers without storing the key on disk or in SQLite.
* When `LLM_API_KEY` is unset on the server and the UI key input is empty, `POST /api/chat` returns `401` and the UI prompts the user to enter their key.
* `docker build -t yt-rag .` builds cleanly under 300 MB, runs on `$PORT` with `READ_ONLY_MODE=true` blocking `/api/sync` (`403`), and serves the baked-in `data/rag.db` snapshot.


#### Step 8: UI Embedding Model Switcher & Prompted Offline Re-Indexing (`yt-rag`)

* **Pre-conditions:** Steps 1–7 completed; `videos.raw_segments_json` is populated in `data/rag.db` from prior playlist syncs.
* **Sub-tasks:**
1. **Track Active Embedding Metadata (`backend/db.py`):**
* Add a `meta(key TEXT PRIMARY KEY, value TEXT)` table in `init_db()` initialized with `embed_model = settings.EMBED_MODEL` and `embed_dim = str(settings.EMBED_DIM)` when `data/rag.db` is first created.
* Add a `reset_vector_tables(new_model: str, new_dim: int)` helper that atomically executes `DROP TABLE IF EXISTS vec_chunks`, `DROP TABLE IF EXISTS chunks_fts`, `DELETE FROM chunks`, recreates `chunks_fts` and `vec_chunks USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{new_dim}])`, and updates `meta`.


2. **Define Curated Embedding Models & Offline Re-Index Stream (`backend/config.py`, `backend/ingest.py`, `backend/main.py`):**
* Define a preset registry of supported embedding models in `backend/config.py`:
* `text-embedding-3-small` (OpenAI, `1536` dims — default)
* `text-embedding-3-large` (OpenAI, `3072` dims)
* `nomic-embed-text` (Ollama local, `768` dims)
* `bge-m3` (Ollama local, `1024` dims)


* Expose `active_embed_model`, `active_embed_dim`, and `available_embed_models` in `GET /api/config`.
* In `backend/ingest.py`, implement `reindex_stored_transcripts(embed_model: str, embed_dim: int, api_key: str | None)` as an async SSE generator:
* Calls `db.reset_vector_tables(embed_model, embed_dim)`.
* Reads all rows from `videos WHERE status = 'indexed'` (`raw_segments_json` is already stored locally—**zero `yt-dlp` or YouTube calls**), re-chunks them, batch-embeds with the selected model, inserts into `chunks` and `vec_chunks`, and yields `{"current": i, "total": n, "video_title": title}` progress events.


* Expose `POST /api/reindex` (accepting `{"embed_model": str}` and optional `X-Api-Key` / `X-Admin-Token` headers). Block with `HTTP 403` when `READ_ONLY_MODE=true` unless valid `X-Admin-Token` is provided.
* In `POST /api/chat`, verify that the requested `embed_model` (if sent) matches `meta.embed_model`; otherwise return `HTTP 409 Conflict` (`"Re-indexing required for selected embedding model"`).


3. **Add Dropdown & Confirmation Prompt Before Querying (`frontend/src/`):**
* In the sidebar settings (hidden or locked to the active badge when `read_only_mode: true`), add an **Embedding Model** `<select>` populated from `GET /api/config`, defaulting to `active_embed_model`.
* **Pre-Query Gate & Confirmation Modal:** Whenever the user selects an embedding model different from `active_embed_model` (or tries to submit a chat query while a mismatch exists):
* Disable the chat input until resolved and display an inline confirmation prompt: *"Switching embeddings to **{model}** requires re-indexing {N} stored transcripts locally (no YouTube re-download). Proceed?"* with **"Re-Index Now"** and **"Cancel"** buttons.
* Clicking **"Cancel"** reverts the dropdown back to `active_embed_model` and re-enables chat immediately.
* Clicking **"Re-Index Now"** streams `POST /api/reindex`, shows a live progress bar (`Re-indexing video X / N...`), updates `active_embed_model` on completion, and unlocks chat.






* **Acceptance Criteria:**
* Changing the embedding dropdown from `text-embedding-3-small` to `text-embedding-3-large` blocks chat submission and displays the confirmation prompt.
* Clicking **"Cancel"** restores the previous model without altering `data/rag.db`.
* Clicking **"Re-Index Now"** rebuilds `vec_chunks` to `3072` dimensions strictly from `videos.raw_segments_json` (verifiable with network access to YouTube disabled) and allows subsequent `POST /api/chat` queries to succeed with accurate timestamp citations.
