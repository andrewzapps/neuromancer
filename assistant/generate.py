from __future__ import annotations

import sys
import time
from ollama import Client
from pydantic import BaseModel, ValidationError

from config import (
    DEFAULT_TOP_K,
    LLM_MODEL,
    MAX_CONTEXT_CHUNK_CHARS,
    MECHANICS_PATH,
    OLLAMA_CHAT_OPTIONS,
    OLLAMA_URL,
    PROMPT_PATH,
)
from retrieve import Candidate, retrieve, warmup
_ollama_client: Client | None = None


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
            f"{chunk.document[:MAX_CONTEXT_CHUNK_CHARS]}"
        )
        chunk_sections.append(section)

    context_block = "\n\n---\n\n".join(chunk_sections)
    user_content = f"## Retrieved context\n\n{context_block}\n\n---\n\n## Question\n{query}"

    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": user_content},
    ]


def _report_validation_error(raw_output: str, exc: ValidationError) -> None:
    print("Failed to validate model output against RAGAnswer schema.", file=sys.stderr)
    print("\nRaw output:\n", raw_output, file=sys.stderr)
    print(f"\nValidation error:\n{exc}", file=sys.stderr)


def _get_ollama_client() -> Client:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = Client(host=OLLAMA_URL)
    return _ollama_client


def _call_ollama(messages: list[dict]) -> str:
    response = _get_ollama_client().chat(
        model=LLM_MODEL,
        messages=messages,
        format=RAGAnswer.model_json_schema(),
        options=OLLAMA_CHAT_OPTIONS,
    )
    return response.message.content


def _parse_answer(raw_output: str) -> RAGAnswer:
    return RAGAnswer.model_validate_json(raw_output)


def generate(query: str, top_k: int = DEFAULT_TOP_K) -> RAGAnswer:
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
            _report_validation_error(raw_output, exc)
            raise


def _print_result(result: RAGAnswer) -> None:
    print(result.answer)
    print()
    print("sources:")
    for source in result.sources:
        print(f"  - {source}")


def _run_interactive() -> None:
    warmup()
    print("Ready. Ask a question (empty line, 'exit', or Ctrl-D to quit).\n")

    while True:
        try:
            user_query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_query or user_query.lower() in ("exit", "quit"):
            break
        try:
            started = time.perf_counter()
            _print_result(generate(user_query))
            elapsed = time.perf_counter() - started
            print(f"\n({elapsed:.1f}s)")
        except ValidationError:
            print("Generation failed, try rephrasing the question.", file=sys.stderr)
        print()


if __name__ == "__main__":
    _run_interactive()
