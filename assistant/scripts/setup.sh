#!/usr/bin/env bash
# Install dependencies, fetch models, and build the search index.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

PROVIDER="${NEUROMANCER_LLM_PROVIDER:-ollama}"
KEY_LATER=0

usage() {
    cat <<'EOF'
Set up NeuroMANCER-GPT.

Usage:
  bash scripts/setup.sh [--api] [options]

Installation modes:

  (default)   Local -- everything runs on your machine, free and offline
              once installed. Downloads ~6.3 GB:
                  nomic-embed-text    274 MB   embeddings
                  bge-reranker-base   1.1 GB   reranking
                  llama3.1:8b         4.9 GB   answers

  --api       API -- answers come from an OpenAI-compatible endpoint, so the
              4.9 GB model is SKIPPED. Downloads ~1.4 GB. Requires
              OPENAI_API_KEY. Ollama is still required: query embeddings
              always run locally.

Options:
  --api                 Shorthand for --provider openai
  --provider PROVIDER   "ollama" (default) or "openai"
  --key-later           In API mode, skip the OPENAI_API_KEY check and enter
                        the key in the app sidebar instead
  --skip-deps           Do not install Python dependencies
  -h, --help            Show this message

Environment:
  OPENAI_API_KEY          API key, required by --api
  OPENAI_BASE_URL         Endpoint override (Azure, vLLM, OpenRouter, ...)
  NEUROMANCER_LLM_MODEL   Override the answering model
  NEUROMANCER_INSTALLER   Set to "uv" to install with uv instead of pip
  NEUROMANCER_ROOT        Repository to index (defaults to this checkout)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --api|--openai)  PROVIDER="openai" ;;
        --provider)      shift; PROVIDER="${1:-}" ;;
        --provider=*)    PROVIDER="${1#*=}" ;;
        --key-later)     KEY_LATER=1 ;;
        --skip-deps)     SKIP_DEPS=1 ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "Unknown option: $1 (try --help)" ;;
    esac
    shift
done

case "${PROVIDER}" in
    ollama|openai) ;;
    *) die "Unknown provider: ${PROVIDER} (expected 'ollama' or 'openai')" ;;
esac

# settings.py reads this to resolve the default model for the chosen provider.
export NEUROMANCER_LLM_PROVIDER="${PROVIDER}"

NEUROMANCER_ROOT="${NEUROMANCER_ROOT:-$(cd "${ASSISTANT_DIR}/.." && pwd)}"

# Before anything is announced, so an unusable interpreter fails immediately.
check_python_version

# Announce the download size
if [ "${PROVIDER}" = "openai" ]; then
    log "Mode: API (OpenAI-compatible endpoint)"
    log "  Will download ~1.4 GB (embeddings + reranker)"
    log "  Skipping llama3.1:8b (4.9 GB) -- answers come from the API"
else
    log "Mode: Local (everything on this machine)"
    log "  Will download ~6.3 GB (embeddings + reranker + llama3.1:8b)"
    log "  For a ~1.4 GB install using your own API key instead, re-run with --api"
fi

# Fail before the download and the index build, not after.
if [ "${PROVIDER}" = "openai" ] && [ "${KEY_LATER}" != "1" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    die "API mode needs an API key, but OPENAI_API_KEY is not set.
      export OPENAI_API_KEY=sk-...
  Then re-run this script. To set the key in the app sidebar instead of the
  environment, re-run with:  bash scripts/setup.sh --api --key-later"
fi

cd "${ASSISTANT_DIR}"

# Installs into the active environment; no virtualenv is created here.
if [ "${SKIP_DEPS:-0}" = "1" ]; then
    log "Skipping dependency install (--skip-deps)"
else
    warn_if_no_env
    read -r -a INSTALL_CMD <<< "$(detect_installer)"
    log "Installing dependencies with ${INSTALL_CMD[*]} into $(python -c 'import sys; print(sys.prefix)')"
    (cd "${NEUROMANCER_ROOT}" && "${INSTALL_CMD[@]}" install -e ".[assistant]") \
        || die "Dependency install failed"
fi

ensure_ollama

# Pulled in both modes: query embeddings always run locally through Ollama.
EMBED_MODEL="$(setting EMBED_MODEL)"
[ -n "${EMBED_MODEL}" ] || die "Could not read EMBED_MODEL from settings.py"
ensure_ollama_model "${EMBED_MODEL}"

if [ "${PROVIDER}" = "ollama" ]; then
    LLM_MODEL="$(setting LLM_MODEL)"
    [ -n "${LLM_MODEL}" ] || die "Could not read LLM_MODEL from settings.py"
    ensure_ollama_model "${LLM_MODEL}"
else
    log "Skipping the local answering model (API mode)"
fi

# retrieve.py loads this with local_files_only=True, so it must already be cached.
log "Ensuring the reranker is cached"
python scripts/fetch_models.py || die "Could not fetch the reranker"

log "Chunking ${NEUROMANCER_ROOT}"
python preprocessing/ingest.py "${NEUROMANCER_ROOT}" || die "Ingestion failed"

# --reset is required: any re-ingest changes the record count, and
# load_vector_store refuses to load into a collection whose count differs.
log "Embedding chunks and building the Chroma + BM25 index (this takes a few minutes)"
python preprocessing/load_vector_store.py --reset || die "Index build failed"

log "Setup complete. Start the app with: bash scripts/run.sh"
