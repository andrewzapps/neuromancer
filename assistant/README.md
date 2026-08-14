# NeuroMANCER-GPT

A retrieval-augmented assistant for the [NeuroMANCER](https://github.com/pnnl/neuromancer)
library. Answers questions about the API, documentation, and examples, grounded in
this repository — chunks the source/docs/examples, embeds them locally, and serves
a chat UI over the resulting index.

## Quick start

The assistant indexes this repository, so it ships with the repo — not with
`pip install neuromancer`. Clone first.

```sh
git clone https://github.com/pnnl/neuromancer.git
cd neuromancer/assistant

bash scripts/setup.sh          # dependencies, models, search index
bash scripts/run.sh            # starts Ollama + the UI at http://localhost:8501
```

<details>
<summary>Windows</summary>

The scripts require a bash shell — use <b>WSL</b> or <b>Git Bash</b>. They will not
run in PowerShell. Run `bash scripts/setup.sh` from Git Bash or WSL.
</details>

Re-run `setup.sh` whenever the NeuroMANCER source, docs, or examples change — it
re-chunks and rebuilds the index from scratch.

## Project layout

```
assistant/
  app/            retrieval, generation, settings, the Streamlit UI
  preprocessing/  chunking (ingest.py) and index building (load_vector_store.py)
  prompts/        system/rewrite prompt templates
  scripts/        setup.sh / run.sh entry points + shared shell helpers
```

## Requirements

| | |
|---|---|
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | required — manages the Python version, venv, and dependencies |
| Python | 3.11+ (fetched automatically by uv if missing) |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~2 GB (models + index) |
| GPU | optional — speeds up local models, not required |
| [Ollama](https://ollama.com/download) | required on both providers (query embeddings) |
| Network | first setup only, to pull models |

Using OpenAI for answers still requires Ollama for embeddings.

