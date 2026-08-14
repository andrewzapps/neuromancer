#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_command uv "Install it from https://docs.astral.sh/uv/getting-started/installation/"

cd "${ASSISTANT_DIR}"

# --- ollama ---------------------------------------------------------------
# required even on the OpenAI provider: query embeddings always go through it
ensure_ollama


missing=()
[ -f "chroma_store/chroma.sqlite3" ] || missing+=("chroma_store/chroma.sqlite3")
[ -f "chroma_store/bm25_index.pkl" ] || missing+=("chroma_store/bm25_index.pkl")
compgen -G "knowledge/*.jsonl" >/dev/null || missing+=("knowledge/*.jsonl")

if [ "${#missing[@]}" -gt 0 ]; then
    die "Search index is missing (${missing[*]}). Run: bash scripts/setup.sh"
fi

# --- launch ---------------------------------------------------------------
log "Starting Streamlit"
if [ -n "${PORT:-}" ]; then
    exec uv run streamlit run app/streamlit_app.py --server.port "${PORT}"
fi
exec uv run streamlit run app/streamlit_app.py
