from __future__ import annotations

import logging
import os
import pickle
import re
from dataclasses import dataclass, replace
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_store"
BM25_INDEX_PATH = "./chroma_store/bm25_index.pkl"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTIONS = ["neuromancer_examples", "neuromancer_docs", "neuromancer_src"]
MAX_QUERY_CHARS = 2000
POOL_SIZE = 24
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
DEFAULT_TOP_K = 5
NOMIC_QUERY_PREFIX = "search_query: "

_client: chromadb.PersistentClient | None = None
_collections: dict[str, chromadb.Collection] | None = None
_bm25_cache: dict[str, Any] | None = None
_reranker: Any | None = None


@dataclass
class Candidate:
    id: str
    document: str
    metadata: dict
    collection_name: str
    dense_dist: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


def _get_embedding_function() -> OllamaEmbeddingFunction:
    return OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _get_collections() -> dict[str, chromadb.Collection]:
    global _collections
    if _collections is None:
        client = _get_client()
        embed_fn = _get_embedding_function()
        _collections = {
            name: client.get_collection(name=name, embedding_function=embed_fn)
            for name in COLLECTIONS
        }
    return _collections


def clean_query(raw: str) -> str:
    cleaned = re.sub(r"\s+", " ", raw.strip())
    if not cleaned:
        raise ValueError("Query is empty")
    if len(cleaned) > MAX_QUERY_CHARS:
        logger.warning(
            "Query exceeds %d characters (%d)",
            MAX_QUERY_CHARS,
            len(cleaned),
        )
        cleaned = cleaned[:MAX_QUERY_CHARS]
    return cleaned


def _prefix_query(cleaned_query: str) -> str:
    return f"{NOMIC_QUERY_PREFIX}{cleaned_query}"


def _handle_ollama_error(exc: BaseException) -> None:
    msg = str(exc).lower()
    if any(s in msg for s in ("connection refused", "failed to connect", "unreachable", "ollama")):
        raise RuntimeError("Ollama unreachable") from exc
    raise exc


def dense_search(query: str, k: int) -> list[Candidate]:
    prefixed = _prefix_query(query)
    collections = _get_collections()
    candidates: list[Candidate] = []

    for collection_name, collection in collections.items():
        try:
            results = collection.query(query_texts=[prefixed], n_results=k)
        except Exception as exc:
            _handle_ollama_error(exc)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            candidates.append(
                Candidate(
                    id=doc_id,
                    document=document or "",
                    metadata=metadata or {},
                    collection_name=collection_name,
                    dense_dist=float(distance) if distance is not None else None,
                )
            )

    candidates.sort(key=lambda c: c.dense_dist if c.dense_dist is not None else float("inf"))
    return candidates


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _live_chunk_count() -> int:
    return sum(collection.count() for collection in _get_collections().values())


def _load_bm25_pickle() -> dict[str, Any] | None:
    if not os.path.exists(BM25_INDEX_PATH):
        return None

    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load BM25 index %s (%s)", BM25_INDEX_PATH, exc)
        return None

    if not isinstance(payload, dict) or "entries" not in payload or "bm25" not in payload:
        logger.warning("BM25 index %s malformed", BM25_INDEX_PATH)
        return None

    expected = payload.get("expected_count")
    if expected is not None and expected != _live_chunk_count():
        logger.warning("BM25 index %s is stale", BM25_INDEX_PATH)
        return None

    return {"entries": payload["entries"], "bm25": payload["bm25"]}


def _get_bm25_index() -> dict[str, Any]:
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    index = _load_bm25_pickle()
    if index is None:
        raise RuntimeError(
            f"BM25 index unavailable at {BM25_INDEX_PATH}. "
        )
    #build cache to save compute
    _bm25_cache = index
    return _bm25_cache


def sparse_search(query: str, k: int) -> list[Candidate]:
    cache = _get_bm25_index()
    bm25 = cache["bm25"]
    entries = cache["entries"]

    #temporary
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    candidates: list[Candidate] = []
    for index in ranked_indices:
        entry = entries[index]
        candidates.append(
            Candidate(
                id=entry["id"],
                document=entry["document"],
                metadata=entry["metadata"],
                collection_name=entry["collection_name"],
                sparse_score=float(scores[index]),
            )
        )
    return candidates


def hybrid_merge(dense: list[Candidate], sparse: list[Candidate]) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    rrf_scores: dict[str, float] = {}

    for rank, candidate in enumerate(dense, start=1):
        rrf_scores[candidate.id] = rrf_scores.get(candidate.id, 0.0) + 1.0 / (RRF_K + rank)
        if candidate.id not in merged:
            merged[candidate.id] = candidate

    for rank, candidate in enumerate(sparse, start=1):
        rrf_scores[candidate.id] = rrf_scores.get(candidate.id, 0.0) + 1.0 / (RRF_K + rank)
        if candidate.id not in merged:
            merged[candidate.id] = candidate
        else:
            existing = merged[candidate.id]
            merged[candidate.id] = replace(
                existing,
                sparse_score=candidate.sparse_score,
            )

    pooled = [
        replace(candidate, rrf_score=rrf_scores[candidate.id])
        for candidate in merged.values()
    ]
    pooled.sort(key=lambda c: c.rrf_score or 0.0, reverse=True)
    return pooled[:POOL_SIZE]


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
            ) from exc
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, candidate.document) for candidate in candidates]
    scores = reranker.predict(pairs)

    ranked = [
        replace(candidate, rerank_score=float(score))
        for candidate, score in zip(candidates, scores)
    ]
    ranked.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
    return ranked[:top_k]


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[Candidate]:
    cleaned = clean_query(query)
    dense = dense_search(cleaned, k=POOL_SIZE)
    sparse = sparse_search(cleaned, k=POOL_SIZE)
    pooled = hybrid_merge(dense, sparse)
    return rerank(cleaned, pooled, top_k=top_k)


def _print_hits(query: str, hits: list[Candidate]) -> None:
    print(f'\nQuery: "{query}"')
    for rank, hit in enumerate(hits, start=1):
        file_path = hit.metadata.get("file_path", "unknown")
        preview = hit.document[: max(1, len(hit.document) * 3 // 4)].replace("\n", " ")
        print(f"  {rank}. {hit.collection_name} | {file_path} | {preview}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    demo_queries = [
        "DictDataset",
    ]

    for demo_query in demo_queries:
        hits = retrieve(demo_query)
        _print_hits(demo_query, hits)
