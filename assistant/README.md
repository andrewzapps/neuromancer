# NeuroMANCER-GPT

A retrieval-augmented assistant for the NeuroMANCER library. Answers questions about
the API, documentation, and examples, grounded in this repository.

## Setup

```sh
bash scripts/setup.sh   # dependencies, models, search index
bash scripts/run.sh     # starts Ollama + the UI at http://localhost:8501
```

Re-run `setup.sh` whenever the NeuroMANCER source, docs, or examples change.

## Requirements

| | |
|---|---|
| Python | 3.11+ |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~2 GB (models + index) |
| GPU | optional — speeds up local models, not required |
| [Ollama](https://ollama.com/download) | required on both providers (query embeddings) |
| Network | first setup only, to pull models |


Using OpenAI for answers still requires Ollama for embeddings.

Provider and model can also be switched from the sidebar at runtime.


