from __future__ import annotations

from collections.abc import Iterator

from ollama import Client
from openai import OpenAI

from settings import (
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_CONTEXT_CHUNK_CHARS,
    MECHANICS_PATH,
    OLLAMA_CHAT_OPTIONS,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PROMPT_PATH,
)
from retrieve import Candidate

_ollama_client: Client | None = None


def _load_system_prompt() -> str:
    base = PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    mechanics = MECHANICS_PATH.read_text(encoding="utf-8").rstrip()
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


def _build_messages(
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

    messages: list[dict] = [{"role": "system", "content": _load_system_prompt()}]
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


def _stream_openai(
    messages: list[dict],
    model: str,
    api_key: str,
) -> Iterator[str]:
    if not api_key:
        raise ValueError(
            "OpenAI API key missing. Set OPENAI_API_KEY or enter it in the sidebar."
        )

    client = OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def _stream_ollama(
    messages: list[dict],
    model: str,
) -> Iterator[str]:
    stream = _get_ollama_client().chat(
        model=model,
        messages=messages,
        options=OLLAMA_CHAT_OPTIONS,
        stream=True,
    )
    for part in stream:
        content = part.message.content
        if content:
            yield content


def stream_from_chunks(
    query: str,
    chunks: list[Candidate],
    history: list[dict] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Iterator[str]:
    messages = _build_messages(query, chunks, history=history)
    provider = (provider or LLM_PROVIDER).strip().lower()
    model = model or LLM_MODEL

    if provider == "openai":
        yield from _stream_openai(
            messages,
            model=model,
            api_key=api_key if api_key is not None else OPENAI_API_KEY,
        )
        return

    yield from _stream_ollama(messages, model=model)
