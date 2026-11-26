#!/usr/bin/env bash
set -e
export OPENSSL_CONF=/dev/null

# Copy .env from .env.example if missing
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — add your LLM_API_KEY or Ollama config."
fi

# Set up virtual environment if not present
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r backend/requirements.txt

# Build frontend if needed (intelligently checks git status and dist presence)
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
  BUILD_REQUIRED=false

  if [ ! -d "frontend/dist" ]; then
    echo "Frontend build missing (frontend/dist not found) — building..."
    BUILD_REQUIRED=true
  elif command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    CURRENT_STATE="$(git log -1 --format=%H -- frontend 2>/dev/null || true):$(git status --porcelain frontend/)"
    PREV_STATE="$(cat frontend/dist/.build_state 2>/dev/null || true)"
    if [ "$CURRENT_STATE" != "$PREV_STATE" ]; then
      echo "Detected changes in frontend/ via git status — rebuilding frontend..."
      BUILD_REQUIRED=true
    else
      echo "Frontend is up to date (no changes detected in frontend/)."
    fi
  fi

  if [ "$BUILD_REQUIRED" = true ]; then
    (cd frontend && npm install --silent && npm run build)
    if [ -d "frontend/dist" ] && command -v git >/dev/null 2>&1; then
      CURRENT_STATE="$(git log -1 --format=%H -- frontend 2>/dev/null || true):$(git status --porcelain frontend/)"
      echo "$CURRENT_STATE" > frontend/dist/.build_state
    fi
  fi
fi

uvicorn backend.main:app --host 0.0.0.0 --port 8000
