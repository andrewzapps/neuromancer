#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NEUROMANCER_ROOT="${NEUROMANCER_ROOT:-$(cd "${ASSISTANT_DIR}/.." && pwd)}"

cd "${ASSISTANT_DIR}"


PYTHON_PATH="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
log "Python: ${PYTHON_PATH}"

# --- dependencies ---------------------------------------------------------
if [ "${SKIP_DEPS:-0}" = "1" ]; then
    log "Skipping dependency install (SKIP_DEPS=1)"
else
    log "Installing dependencies from requirements.txt"
    "${PYTHON_BIN}" -m pip install -q -r requirements.txt || die "Dependency install failed"
fi

# --- 3. ollama ------------------------------------------------------------
# needed now, not just at run time: ingestion embeds every chunk through it
ensure_ollama

# --- 4. embedding model ---------------------------------------------------
EMBED_MODEL="$(setting EMBED_MODEL)"
[ -n "${EMBED_MODEL}" ] || die "Could not read EMBED_MODEL from settings.py"
ensure_ollama_model "${EMBED_MODEL}"

# --- 5. reranker ----------------------------------------------------------
# retrieve.py loads this with local_files_only=True, so it must already be cached
log "Ensuring the reranker is cached"
"${PYTHON_BIN}" scripts/fetch_models.py || die "Could not fetch the reranker"

# --- 6. chunk the repository ---------------------------------------------
log "Chunking ${NEUROMANCER_ROOT}"
"${PYTHON_BIN}" preprocessing/ingest.py "${NEUROMANCER_ROOT}" || die "Ingestion failed"

# --- 7. build the index ---------------------------------------------------
# --reset is required: load_vector_store refuses to load into a collection whose
# existing count differs from the new record count, which any re-ingest produces
log "Embedding chunks and building the Chroma + BM25 index (this takes a few minutes)"
"${PYTHON_BIN}" preprocessing/load_vector_store.py --reset || die "Index build failed"

log "Setup complete. Start the app with: bash scripts/run.sh"
