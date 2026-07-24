from __future__ import annotations

from collections.abc import Iterator

from ollama import Client
from pydantic import BaseModel

from config import (
    DEFAULT_TOP_K,
    LLM_MODEL,
    MAX_CONTEXT_CHUNK_CHARS,
    MECHANICS_PATH,
    OLLAMA_CHAT_OPTIONS,
    OLLAMA_URL,
    PROMPT_PATH,
)
from retrieve import Candidate, retrieve

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


def sources_from_chunks(chunks: list[Candidate]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        path = chunk.metadata.get("file_path")
        if not path or path == "unknown" or path in seen:
            continue
        seen.add(path)
        sources.append(path)
    return sources


def build_messages(
    query: str,
    chunks: list[Candidate],
    history: list[dict] | None = None,
) -> list[dict]:
    chunk_sections: list[str] = []

    for chunk in chunks:
        file_path = chunk.metadata.get("file_path", "unknown")
        symbol_name = chunk.metadata.get("symbol_name")
        header = f"### {file_path}"
        if symbol_name and symbol_name != "unknown":
            header += f" — {symbol_name}"
        section = f"{header}\n\n{chunk.document[:MAX_CONTEXT_CHUNK_CHARS]}"
        chunk_sections.append(section)

    context_block = "\n\n---\n\n".join(chunk_sections)
    user_content = (
        f"## Reference excerpts from the NeuroMANCER repository\n\n"
        f"{context_block}\n\n---\n\n## Question\n{query}"
    )

    messages: list[dict] = [{"role": "system", "content": load_system_prompt()}]
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


def _get_ollama_client() -> Client:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = Client(host=OLLAMA_URL)
    return _ollama_client


def _call_ollama(messages: list[dict]) -> str:
    response = _get_ollama_client().chat(
        model=LLM_MODEL,
        messages=messages,
        options=OLLAMA_CHAT_OPTIONS,
    )
    return response.message.content or ""


def stream_from_chunks(
    query: str,
    chunks: list[Candidate],
    history: list[dict] | None = None,
) -> Iterator[str]:
    messages = build_messages(query, chunks, history=history)
    stream = _get_ollama_client().chat(
        model=LLM_MODEL,
        messages=messages,
        options=OLLAMA_CHAT_OPTIONS,
        stream=True,
    )
    for part in stream:
        content = part.message.content
        if content:
            yield content


def generate_from_chunks(
    query: str,
    chunks: list[Candidate],
    history: list[dict] | None = None,
) -> RAGAnswer:
    messages = build_messages(query, chunks, history=history)
    answer = _call_ollama(messages)
    return RAGAnswer(answer=answer, sources=sources_from_chunks(chunks))


def generate(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    history: list[dict] | None = None,
) -> RAGAnswer:
    chunks = retrieve(query, top_k=top_k)
    return generate_from_chunks(query, chunks, history=history)
