"""FastAPI backend application serving RAG endpoints and static frontend."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import backend.db as db
import backend.ingest as ingest
import backend.rag as rag
from backend.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context initializing database on server startup."""
    db.init_db()
    yield


app = FastAPI(
    title="yt-rag",
    version="0.1.0",
    description="YouTube Playlist RAG Engine with Timestamp Lookup, Opinion Synthesis, and Timeline Evolution",
    lifespan=lifespan,
)

# Enable CORS for local dev server (e.g. Vite on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SyncRequest(BaseModel):
    playlist_ids: list[str] | None = None
    max_videos_per_playlist: int | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to synthesize")
    playlist_id: str | None = Field(default=None, description="Optional playlist scope")
    mode: Literal["auto", "lookup", "opinion", "evolution"] = Field(
        default="auto",
        description="Retrieval & synthesis mode",
    )


@app.get("/health")
def health_check():
    """Health status endpoint."""
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    """Expose read-only mode and server key availability."""
    return {
        "read_only_mode": settings.READ_ONLY_MODE,
        "has_server_key": bool(settings.LLM_API_KEY),
    }


@app.get("/api/playlists")
def get_playlists(
    check_remote: bool = False,
    x_admin_token: str | None = Header(default=None),
):
    """Return configured playlists and video sync statistics."""
    if check_remote:
        if settings.READ_ONLY_MODE and (not settings.ADMIN_TOKEN or x_admin_token != settings.ADMIN_TOKEN):
            raise HTTPException(status_code=403, detail="Remote playlist check is disabled in read-only mode")
        # Check YouTube for new unindexed videos
        updates = ingest.check_playlist_updates()
        return updates

    # Return local database statistics
    stats = db.get_playlist_stats()
    return stats


@app.get("/api/sync/status")
def sync_status():
    """Return rate limit and sync status."""
    return ingest.get_rate_limit_info()


@app.post("/api/sync/clear-rate-limit")
def clear_sync_rate_limit():
    """Clear rate limit flag."""
    ingest.clear_rate_limited()
    return {"status": "ok"}


@app.post("/api/sync")
async def sync_playlists_endpoint(
    req: SyncRequest | None = None,
    x_admin_token: str | None = Header(default=None),
):
    """Trigger incremental sync, streaming Server-Sent Events (SSE) progress."""
    if settings.READ_ONLY_MODE and (not settings.ADMIN_TOKEN or x_admin_token != settings.ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Sync is disabled in read-only mode")

    playlist_ids = req.playlist_ids if req else None
    max_videos = req.max_videos_per_playlist if req else None

    async def sse_event_stream() -> AsyncGenerator[str, None]:
        was_rate_limited = False
        try:
            async for event in ingest.sync_playlists(
                playlist_ids=playlist_ids,
                max_videos_per_playlist=max_videos,
            ):
                if event.get("status") == "rate_limited":
                    was_rate_limited = True
                yield f"data: {json.dumps(event)}\n\n"

            if not was_rate_limited:
                yield f"data: {json.dumps({'status': 'done', 'message': 'Sync completed successfully'})}\n\n"
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate limit" in err_msg.lower() or "blocked" in err_msg.lower():
                ingest.set_rate_limited(err_msg)
                yield f"data: {json.dumps({'status': 'rate_limited', 'message': f'Rate limit exceeded: {err_msg}'})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'error', 'message': err_msg})}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat_endpoint(
    req: ChatRequest,
    x_provider: str | None = Header(default="openai"),
    x_api_key: str | None = Header(default=None),
):
    """Multi-mode RAG query endpoint returning synthesized answer and timestamped sources."""
    provider = (x_provider or "openai").lower().strip()
    if provider not in settings.CHAT_PROVIDERS:
        provider = "openai"

    request_api_key = (x_api_key or "").strip()
    chat_key = request_api_key or settings.LLM_API_KEY
    if not chat_key:
        raise HTTPException(
            status_code=401,
            detail="Enter an API key in the sidebar to ask questions",
        )

    # Resolve embedding key: server EMBED_API_KEY -> LLM_API_KEY -> user's OpenAI key if using openai
    embed_key = settings.EMBED_API_KEY or settings.LLM_API_KEY or (request_api_key if provider == "openai" else "")
    if not embed_key:
        raise HTTPException(
            status_code=401,
            detail="No OpenAI embedding key configured on server. Please provide an OpenAI API key or configure EMBED_API_KEY.",
        )

    try:
        response = await rag.query_rag(
            question=req.question,
            playlist_id=req.playlist_id,
            mode=req.mode,
            provider=provider,
            chat_key=chat_key,
            embed_key=embed_key,
        )
        return response
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Incorrect API key" in err_msg or "invalid_api_key" in err_msg:
            raise HTTPException(status_code=401, detail=f"Invalid API key for {provider}: {err_msg}")
        raise HTTPException(status_code=500, detail=err_msg)


# Mount built frontend dist files if available
FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "message": "yt-rag API server running. Frontend dist not built yet. Run Step 6 build to serve the UI.",
            "docs": "/docs",
            "playlists": "/api/playlists",
        }
