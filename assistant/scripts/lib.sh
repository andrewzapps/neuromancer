#!/usr/bin/env bash

ASSISTANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_LOG="${OLLAMA_LOG:-${TMPDIR:-/tmp}/neuromancer-gpt-ollama.log}"
OLLAMA_START_TIMEOUT="${OLLAMA_START_TIMEOUT:-30}"

log()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror\033[0m %s\n' "$*" >&2; exit 1; }

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is not installed or not on PATH. $2"
}

ollama_is_up() {
    curl -sf -m 3 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

# Start ollama serve only if it is not already answering. Safe to call repeatedly:
# setup.sh and run.sh both need it, and a developer may already have one running.
ensure_ollama() {
    if ollama_is_up; then
        log "Ollama already running at ${OLLAMA_URL}"
        return 0
    fi

    require_command ollama "Install it from https://ollama.com/download"

    log "Starting Ollama (log: ${OLLAMA_LOG})"
    OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}" \
    OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
        nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &

    local waited=0
    until ollama_is_up; do
        sleep 1
        waited=$((waited + 1))
        if [ "${waited}" -ge "${OLLAMA_START_TIMEOUT}" ]; then
            die "Ollama did not become healthy within ${OLLAMA_START_TIMEOUT}s. See ${OLLAMA_LOG}"
        fi
    done
    log "Ollama ready after ${waited}s"
}

# Pull an ollama model only when it is missing
ensure_ollama_model() {
    local model="$1"
    if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -q "^${model}\(:latest\)\?$"; then
        log "Ollama model present: ${model}"
        return 0
    fi
    log "Pulling Ollama model: ${model}"
    ollama pull "${model}" || die "Failed to pull ${model}"
}


setting() {
    (cd "${ASSISTANT_DIR}/app" && uv run python -c "import settings; print(getattr(settings, '$1'))")
}
