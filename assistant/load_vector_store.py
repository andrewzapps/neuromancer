#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_PATH = "./chroma_store"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 100

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


def add_batches(collection, records: list[dict]) -> None:
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        collection.add(
            ids=[chroma_id(r) for r in batch],
            documents=[r["content"] for r in batch],
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
    add_batches(collection, records)
    return collection


def run_test_query(collection: chromadb.Collection) -> None:
    print('\nTest query on neuromancer_examples: "how do I create a training dataset?"')
    results = collection.query(
        query_texts=["how do I create a training dataset?"],
        n_results=3,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for rank, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        file_path = (meta or {}).get("file_path", "unknown")
        text = doc or ""
        preview = text[: len(text) // 2].replace("\n", " ")
        print(f"  {rank}. {file_path}")
        print(f"     {preview}")


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

    run_test_query(loaded["neuromancer_examples"])


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
