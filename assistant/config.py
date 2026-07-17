"""Shared paths and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

ASSISTANT_DIR = Path(__file__).resolve().parent

KNOWLEDGE_DIR = ASSISTANT_DIR / "knowledge"
PROMPTS_DIR = ASSISTANT_DIR / "prompts"
PROMPT_PATH = PROMPTS_DIR / "prompt.txt"
MECHANICS_PATH = PROMPTS_DIR / "mechanics.txt"

CHROMA_PATH = str(ASSISTANT_DIR / "chroma_store")
BM25_INDEX_PATH = str(ASSISTANT_DIR / "chroma_store" / "bm25_index.pkl")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = os.environ.get("NEUROMANCER_LLM_MODEL", "llama3.1:8b")
OLLAMA_CHAT_OPTIONS = {"num_ctx": 4096, "num_predict": 400}

COLLECTION_NAMES = ["neuromancer_examples", "neuromancer_docs", "neuromancer_src"]
KNOWLEDGE_COLLECTIONS = [
    (KNOWLEDGE_DIR / "examples.jsonl", "neuromancer_examples"),
    (KNOWLEDGE_DIR / "docs.jsonl", "neuromancer_docs"),
    (KNOWLEDGE_DIR / "src.jsonl", "neuromancer_src"),
]

DEFAULT_TOP_K = int(os.environ.get("NEUROMANCER_TOP_K", "3"))
MAX_CONTEXT_CHUNK_CHARS = 2500

BATCH_SIZE = 100
DOC_PREFIX = "search_document: "
