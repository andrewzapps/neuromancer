from __future__ import annotations

import os
import sys
from pathlib import Path
from ollama import Client
from pydantic import BaseModel, ValidationError
from retrieve import Candidate, retrieve

MODEL = os.environ.get("NEUROMANCER_LLM_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
PROMPT_PATH = KNOWLEDGE_DIR / "prompt.txt"
MECHANICS_PATH = KNOWLEDGE_DIR / "mechanics.txt"


class RAGAnswer(BaseModel):
    answer: str
    sources: list[str]


def load_mechanics() -> str:
    return MECHANICS_PATH.read_text(encoding="utf-8").rstrip()


def load_system_prompt() -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    mechanics = load_mechanics()
    return f"{base}\n\n{mechanics}"


def build_messages(query: str, chunks: list[Candidate]) -> list[dict]:
    chunk_sections: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        file_path = chunk.metadata.get("file_path", "unknown")
        symbol_name = chunk.metadata.get("symbol_name", "unknown")
        section = (
            f"### Chunk {index}\n"
            f"- collection: {chunk.collection_name}\n"
            f"- file_path: {file_path}\n"
            f"- symbol: {symbol_name}\n\n"
            f"{chunk.document}"
        )
        chunk_sections.append(section)

    context_block = "\n\n---\n\n".join(chunk_sections)
    user_content = f"## Retrieved context\n\n{context_block}\n\n---\n\n## Question\n{query}"

    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": user_content},
    ]


def _exit_validation_error(raw_output: str, exc: ValidationError) -> None:
    print("Failed to validate model output against RAGAnswer schema.", file=sys.stderr)
    print("\nRaw output:\n", raw_output, file=sys.stderr)
    print(f"\nValidation error:\n{exc}", file=sys.stderr)
    sys.exit(1)


def _call_ollama(messages: list[dict]) -> str:
    client = Client(host=OLLAMA_URL)
    response = client.chat(
        model=MODEL,
        messages=messages,
        format=RAGAnswer.model_json_schema(),
    )
    return response.message.content


def _parse_answer(raw_output: str) -> RAGAnswer:
    return RAGAnswer.model_validate_json(raw_output)


def generate(query: str, top_k: int = 5) -> RAGAnswer:
    chunks = retrieve(query, top_k=top_k)
    messages = build_messages(query, chunks)

    raw_output = _call_ollama(messages)
    try:
        return _parse_answer(raw_output)
    except ValidationError:
        raw_output = _call_ollama(messages)
        try:
            return _parse_answer(raw_output)
        except ValidationError as exc:
            _exit_validation_error(raw_output, exc)


def _print_result(result: RAGAnswer) -> None:
    print(result.answer)
    print()
    print("sources:")
    for source in result.sources:
        print(f"  - {source}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:]).strip()
    else:
        user_query = input("Question: ").strip()

    if not user_query:
        print("Query is empty.", file=sys.stderr)
        sys.exit(1)

    answer = generate(user_query)
    _print_result(answer)
