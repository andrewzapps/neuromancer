#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_command uv "Install it from https://docs.astral.sh/uv/getting-started/installation/"

NEUROMANCER_ROOT="${NEUROMANCER_ROOT:-$(cd "${ASSISTANT_DIR}/.." && pwd)}"

cd "${ASSISTANT_DIR}"

# --- dependencies ---------------------------------------------------------
if [ "${SKIP_DEPS:-0}" = "1" ]; then
    log "Skipping dependency install (SKIP_DEPS=1)"
else
    log "Syncing dependencies (uv)"
    uv sync || die "Dependency install failed"
fi


ensure_ollama

# --- embedding model ---------------------------------------------------
EMBED_MODEL="$(setting EMBED_MODEL)"
[ -n "${EMBED_MODEL}" ] || die "Could not read EMBED_MODEL from settings.py"
ensure_ollama_model "${EMBED_MODEL}"


LLM_PROVIDER="$(setting LLM_PROVIDER)"
if [ "${LLM_PROVIDER}" = "ollama" ]; then
    LLM_MODEL="$(setting LLM_MODEL)"
    [ -n "${LLM_MODEL}" ] || die "Could not read LLM_MODEL from settings.py"
    ensure_ollama_model "${LLM_MODEL}"
fi

# --- reranker ----------------------------------------------------------
# retrieve.py loads this with local_files_only=True, so it must already be cached
log "Ensuring the reranker is cached"
uv run python scripts/fetch_models.py || die "Could not fetch the reranker"

# --- chunking--------------------------------------------
log "Chunking ${NEUROMANCER_ROOT}"
uv run python preprocessing/ingest.py "${NEUROMANCER_ROOT}" || die "Ingestion failed"

# ---build the index ---------------------------------------------------
# --reset is required: load_vector_store refuses to load into a collection whose
# existing count differs from the new record count, which any re-ingest produces
log "Embedding chunks and building the Chroma + BM25 index (this takes a few minutes)"
uv run python preprocessing/load_vector_store.py --reset || die "Index build failed"

log "Setup complete. Start the app with: bash scripts/run.sh"
