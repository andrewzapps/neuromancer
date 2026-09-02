# NeuroMANCER-GPT

A retrieval-augmented assistant for [NeuroMANCER](https://github.com/pnnl/neuromancer).
It answers questions about the library's API, documentation, and examples, grounded in
this repository — it chunks the source, docs, and notebooks, embeds them into a local
search index, and serves a chat UI over the result.


---

## Prerequisites

### 1. Ollama (required)

[**Install Ollama**](https://ollama.com/download).

```sh
brew install ollama                              # macOS
curl -fsSL https://ollama.com/install.sh | sh    # Linux
winget install Ollama.Ollama                     # Windows (PowerShell)
```

Check it is on your PATH before continuing:

```sh
ollama --version
```

Ollama is required in **both** installation modes, including the API mode below. Query
embeddings always run locally through Ollama; only the *answers* can be delegated to an
API. The setup and run scripts start `ollama serve` for you if it is not already running.

### 2. Everything else

| | |
|---|---|
| Python | 3.11 or newer (NeuroMANCER itself supports 3.9+, but the assistant needs 3.11+) |
| Shell | bash — on Windows use **WSL** or **Git Bash**; the scripts do not run in PowerShell |
| RAM | 8 GB minimum, 16 GB recommended |
| GPU | optional; speeds up local models but is not required |
| Network | first setup only, to download models |

---

## Choose your install

The only meaningful difference is the 4.9 GB answering model.

| | **Local** (default) | **API** (`--api`) |
|---|---|---|
| Ollama runtime | required | **required** (embeddings) |
| `nomic-embed-text` — embeddings | 274 MB | 274 MB |
| `bge-reranker-base` — reranking | 1.1 GB | 1.1 GB |
| `llama3.1:8b` — answers | **4.9 GB** | **skipped** |
| `OPENAI_API_KEY` | not needed | **required** |
| **First-run download** | **~6.3 GB** | **~1.4 GB** |
| Running cost | free, offline after setup | per-token, needs network |

Pick **Local** if you want everything on your machine and don't mind the download.
Pick **API** if you already have an OpenAI key and would rather not store a 5 GB model.

---

## Install


```sh
git clone https://github.com/pnnl/neuromancer.git
cd neuromancer

# activate whichever environment you keep neuromancer in, e.g.
conda activate neuromancer
```

Then run the setup for your chosen mode:

```sh
# Local — everything on this machine, ~6.3 GB
bash assistant/scripts/setup.sh

# API — bring your own key, ~1.4 GB
export OPENAI_API_KEY=sk-...
bash assistant/scripts/setup.sh --api
```

`setup.sh` installs dependencies, downloads the models for your mode, chunks the
repository, and builds the search index. It prints what it is about to download before it
starts, so you can back out.

To enter your API key in the app's sidebar instead of the environment, use
`bash assistant/scripts/setup.sh --api --key-later`.

<details>
<summary>Installing dependencies yourself</summary>

`setup.sh` installs with pip by default, which is what NeuroMANCER uses everywhere
else and is present in every environment. To use [uv](https://docs.astral.sh/uv/)
instead — it works and is considerably faster — set `NEUROMANCER_INSTALLER=uv`:

```sh
bash assistant/scripts/setup.sh                             # pip (default)
NEUROMANCER_INSTALLER=uv bash assistant/scripts/setup.sh    # uv
```

Either way, the install step is the equivalent of running this from the repository
root, into your active environment:

```sh
pip install -e ".[assistant]"        # or: uv pip install -e ".[assistant]"
```

Skip the step entirely with `bash assistant/scripts/setup.sh --skip-deps`.
</details>

---

## Run

```sh
bash assistant/scripts/run.sh
```

This starts Ollama if needed and opens the UI at <http://localhost:8501>. Set `PORT` to
use a different port.

Re-run `setup.sh` whenever the NeuroMANCER source, docs, or examples change — it re-chunks
and rebuilds the index from scratch.

---

## Configuration

All configuration is via environment variables; defaults live in `app/settings.py`.

| Variable | Default | Purpose |
|---|---|---|
| `NEUROMANCER_LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `NEUROMANCER_LLM_MODEL` | `llama3.1:8b` / `gpt-4o-mini` | Answering model |
| `OPENAI_API_KEY` | — | Required when the provider is `openai` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Point at any OpenAI-compatible endpoint (Azure, vLLM, OpenRouter, LM Studio) |
| `NEUROMANCER_TOP_K` | `8` | Chunks passed to the model as context |
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is listening |
| `NEUROMANCER_ROOT` | this checkout | Repository to index |
| `PORT` | `8501` | Streamlit port |
| `NEUROMANCER_INSTALLER` | `pip` | Set to `uv` to install with uv instead |

Provider and model can also be switched at runtime from the app's sidebar.

---

## Project layout

```
assistant/
  app/            retrieval, generation, settings, the Streamlit UI
  preprocessing/  chunking (ingest.py) and index building (load_vector_store.py)
  prompts/        system/rewrite prompt templates
  scripts/        setup.sh / run.sh entry points + shared shell helpers
  .streamlit/     Streamlit theme and server config
```

Two directories are generated by `setup.sh` and are not tracked in git:
`knowledge/` (the chunked corpus, as JSONL) and `chroma_store/` (the Chroma vector
database and the BM25 index).

`.streamlit/` is a dot-directory because Streamlit only reads its configuration from that
exact path — it is a framework convention, not a preference.

---

## Troubleshooting

**`ollama is not installed or not on PATH`** — install it with the command for your OS
above. The assistant cannot run without it, even in API mode.

**`The assistant needs Python 3.11+`** — your active environment is older. NeuroMANCER
supports 3.9+, so this may be deliberate; create a separate environment for the assistant:

```sh
conda create -n neuromancer-gpt python=3.11 -y && conda activate neuromancer-gpt
```

**`Search index is missing`** — `run.sh` was called before `setup.sh` finished. Re-run
`bash assistant/scripts/setup.sh`.

**`No virtual environment detected`** — you are about to install into the system Python.
Activate a conda or venv environment first, or confirm the prompt if that is what you want.

**Answers are slow on the local model** — `llama3.1:8b` needs roughly 8 GB of RAM. Either
use `--api` mode, or set `NEUROMANCER_LLM_MODEL` to something smaller such as
`llama3.2:3b`.
