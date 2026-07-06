#!/usr/bin/env python3

import argparse
import json
import pickle
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from retrieve import BM25_INDEX_PATH, _tokenize

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_PATH = "./chroma_store"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 100


DOC_PREFIX = "search_document: "

METADATA_FIELDS = ("file_path", "symbol_name", "start_line", "end_line", "source_type")

COLLECTIONS = [
    (KNOWLEDGE_DIR / "examples.jsonl", "neuromancer_examples", 847),
    (KNOWLEDGE_DIR / "docs.jsonl", "neuromancer_docs", 110),
    (KNOWLEDGE_DIR / "src.jsonl", "neuromancer_src", 1996),
]

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("id", "content", "file_path", "source_type"):
                if key not in record:
                    raise ValueError(f"{path}:{line_no} missing required field {key!r}")
            records.append(record)
    return records


def chroma_id(record: dict) -> str:
    #creates unique chroma id for api or impl
    source_type = record["source_type"]
    base_id = record["id"]
    if source_type in ("api", "impl"):
        return f"{base_id}:{source_type}"
    return base_id


def build_metadata(record: dict) -> dict:
    metadata = {}
    for key in METADATA_FIELDS:
        value = record.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def get_embedding_function() -> OllamaEmbeddingFunction:
    return OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL)


def collection_exists(client: chromadb.PersistentClient, name: str) -> bool:
    return name in {c.name for c in client.list_collections()}


def add_batches(collection, records: list[dict], embed_fn: OllamaEmbeddingFunction) -> None:
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        documents = [r["content"] for r in batch]
        # embed the prefixed text, but store the unprefixed document.
        embeddings = embed_fn([DOC_PREFIX + doc for doc in documents])
        collection.add(
            ids=[chroma_id(r) for r in batch],
            embeddings=embeddings,
            documents=documents,
            metadatas=[build_metadata(r) for r in batch],
        )


def load_collection(
    client: chromadb.PersistentClient,
    name: str,
    records: list[dict],
    expected_count: int,
    embed_fn: OllamaEmbeddingFunction,
    reset: bool,
) -> chromadb.Collection:
    if reset and collection_exists(client, name):
        client.delete_collection(name)

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
    )
    current_count = collection.count()

    if not reset and current_count == expected_count:
        print(f"Skipping {name}: already loaded ({current_count} records)")
        return collection

    if not reset and current_count > 0 and current_count != expected_count:
        print(
            f"Error: {name} has {current_count} records, expected {expected_count}. "
            "Re-run with --reset to rebuild.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if len(records) != expected_count:
        print(
            f"Warning: {name} JSONL has {len(records)} records, expected {expected_count}",
            file=sys.stderr,
        )

    if reset and current_count > 0:
        client.delete_collection(name)
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=embed_fn,
        )

    print(f"Loading {name} ({len(records)} records)...")
    add_batches(collection, records, embed_fn)
    return collection


def build_and_save_bm25_index(client: chromadb.PersistentClient) -> None:
    from rank_bm25 import BM25Okapi

    entries: list[dict] = []
    for _, collection_name, _ in COLLECTIONS:
        collection = client.get_collection(name=collection_name)
        data = collection.get(include=["documents", "metadatas"])
        for doc_id, document, metadata in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
        ):
            entries.append(
                {
                    "id": doc_id,
                    "document": document or "",
                    "metadata": metadata or {},
                    "collection_name": collection_name,
                }
            )

    tokenized_corpus = [_tokenize(entry["document"]) for entry in entries]
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "entries": entries,
        "bm25": bm25,
        "expected_count": len(entries),
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"Built BM25 index: {len(entries)} chunks -> {BM25_INDEX_PATH}")


def main(reset: bool = False) -> None:
    embed_fn = get_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    loaded = {}
    for jsonl_path, collection_name, expected_count in COLLECTIONS:
        records = read_jsonl(jsonl_path)
        loaded[collection_name] = load_collection(
            client,
            collection_name,
            records,
            expected_count,
            embed_fn,
            reset,
        )

    print()
    for collection_name, collection in loaded.items():
        print(f"{collection_name}: {collection.count()}")

    build_and_save_bm25_index(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load ingested JSONL chunks into a local ChromaDB vector store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild collections before loading.",
    )
    args = parser.parse_args()
    main(reset=args.reset)
