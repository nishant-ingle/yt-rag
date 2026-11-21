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

# Build frontend if frontend package exists and dist is not yet built
if [ -d "frontend" ] && [ -f "frontend/package.json" ] && [ ! -d "frontend/dist" ]; then
  (cd frontend && npm install --silent && npm run build)
fi

uvicorn backend.main:app --host 0.0.0.0 --port 8000
