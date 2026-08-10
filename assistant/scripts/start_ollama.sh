#!/usr/bin/env bash
#
# Start Ollama on its own. Kept for convenience -- setup.sh and run.sh both call
# ensure_ollama themselves, so you rarely need this directly.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ensure_ollama
log "Ollama is serving at ${OLLAMA_URL} (log: ${OLLAMA_LOG})"
