from __future__ import annotations

import re
import time

import streamlit as st

from generate import (
    RewriteResult,
    contextualize_query,
    sources_from_chunks,
    stream_from_chunks,
)
from retrieve import retrieve, warmup
from settings import (
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODELS,
)

st.set_page_config(page_title="NeuroMANCER-GPT")

_CODE_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_DISPLAY_BRACKETS = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_INLINE_PARENS = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_BRACKET_TEX = re.compile(
    r"\[\s*((?:[^\[\]]|\n)*?\\[A-Za-z]+(?:[^\[\]]|\n)*?)\s*\]",
    re.DOTALL,
)

_STRAY_DOUBLE_DOLLAR = re.compile(r"(?<!\n)\$\$(?!\n)")


def _is_alone_on_line(text: str, start: int, end: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return not text[line_start:start].strip() and not text[end:line_end].strip()


def _to_math(match: re.Match, *, only_when_alone: bool = False) -> str:
    expression = match.group(1).strip()
    if _is_alone_on_line(match.string, match.start(), match.end()):
        return f"$$\n{expression}\n$$"
    if only_when_alone:
        return match.group(0)
    return f"${expression}$"


def sanitize_markdown(text: str) -> str:
    if not text:
        return text

    parts = _CODE_FENCE.split(text)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        part = _DISPLAY_BRACKETS.sub(_to_math, part)
        part = _INLINE_PARENS.sub(lambda m: f"${m.group(1).strip()}$", part)
        part = _BRACKET_TEX.sub(lambda m: _to_math(m, only_when_alone=True), part)
        part = _STRAY_DOUBLE_DOLLAR.sub("$ $", part)
        out.append(part)
    return "".join(out)


st.markdown(
    """
<style>
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none !important;
}
[data-testid="stChatMessage"] > div:first-child:has(
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"]
) {
    width: 0 !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"] {
    width: 100% !important;
    max-width: 100% !important;
}
[data-testid="stChatInput"] > div {
    border-radius: 1.5rem !important;
    min-height: 3.25rem !important;
    max-height: 12rem !important;
    height: auto !important;
    padding: 0.35rem 0.45rem 0.35rem 1.4rem !important;
    align-items: flex-end !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"] [data-testid="stChatInputTextArea"],
[data-testid="stChatInput"] textarea {
    border-radius: 1.25rem !important;
    min-height: 2.5rem !important;
    max-height: 10.5rem !important;
    height: auto !important;
    padding: 0.65rem 0 0.65rem 0.35rem !important;
    line-height: 1.5 !important;
    font-size: 1rem !important;
    caret-color: #2563eb !important;
    text-indent: 0.15rem !important;
    resize: none !important;
    overflow-y: auto !important;
    field-sizing: content;
}
[data-testid="stChatInput"] textarea:focus::placeholder,
[data-testid="stChatInput"] [data-testid="stChatInputTextArea"]:focus::placeholder {
    color: transparent !important;
    opacity: 0 !important;
}
[data-testid="stChatInput"] > div > div {
    align-items: flex-end !important;
    height: auto !important;
    min-height: 2.5rem !important;
    max-height: 10.5rem !important;
}
[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInput"] button {
    border-radius: 9999px !important;
    width: 2rem !important;
    height: 2rem !important;
    min-height: 2rem !important;
    padding: 0 !important;
    margin: 0 0 0.25rem 0 !important;
    align-self: flex-end !important;
}
/* Slightly darker code blocks in light mode */
[data-testid="stMarkdownContainer"] pre,
[data-testid="stCode"],
div[data-testid="stCodeBlock"] {
    background-color: #eceef2 !important;
}
[data-testid="stMarkdownContainer"] pre code,
[data-testid="stCode"] code,
div[data-testid="stCodeBlock"] code {
    background-color: transparent !important;
}
[data-testid="stMarkdownContainer"] code:not(pre code) {
    background-color: #e4e7ec !important;
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def init_rag():
    warmup()
    return True


@st.cache_data(ttl=1800, show_spinner=False)
def retrieve_and_rerank(query: str):
    return retrieve(query)


def build_status_caption(
    research_s: float,
    stream_s: float,
    rewrite: RewriteResult,
) -> str:
    #timings and rewritten query
    caption = f"research {research_s:.1f}s  ·  stream {stream_s:.1f}s"
    if rewrite.error:
        return (
            f"{caption}  ·  ⚠ query rewrite unavailable ({rewrite.error}) — "
            "searched your question as typed"
        )
    if rewrite.rewritten:
        return f'{caption}  ·  searched: "{rewrite.query}"'
    return caption


def get_recent_context(n: int = 3) -> list[dict]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-n:]
    ]


def _init_llm_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "provider_choice" not in st.session_state:
        st.session_state.provider_choice = (
            "openai" if LLM_PROVIDER == "openai" else "ollama"
        )
    if "openai_model_choice" not in st.session_state:
        st.session_state.openai_model_choice = (
            LLM_MODEL if LLM_MODEL in OPENAI_MODELS else OPENAI_MODELS[0]
        )
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = (
            LLM_MODEL if LLM_PROVIDER == "ollama" else "llama3.1:8b"
        )
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = OPENAI_API_KEY


@st.fragment
def llm_sidebar() -> None:
    st.caption("LLM")
    provider = st.selectbox(
        "Provider",
        options=["ollama", "openai"],
        format_func=lambda p: "Ollama" if p == "ollama" else "OpenAI",
        key="provider_choice",
    )
    if provider == "openai":
        model = st.selectbox(
            "Model",
            options=OPENAI_MODELS,
            format_func=lambda m: (
                "Mini (gpt-4o-mini)" if m == "gpt-4o-mini" else "Sol (gpt-5.6-sol)"
            ),
            key="openai_model_choice",
        )
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            key="openai_api_key",
        )
    else:
        model = st.text_input("Model", key="ollama_model")
        api_key = None

    st.session_state.active_provider = provider
    st.session_state.active_model = model
    st.session_state.active_api_key = api_key
    st.caption(f"Active: `{provider}` / `{model}`")

    # clear chat on provider
    llm_sig = (provider, model if provider == "openai" else "ollama")
    prev_sig = st.session_state.get("_llm_sig")
    if prev_sig is not None and llm_sig != prev_sig:
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state._llm_sig = llm_sig
        st.rerun(scope="app")
    st.session_state._llm_sig = llm_sig


# -----------------------------------------------------------------------------

init_rag()
_init_llm_session_state()
with st.sidebar:
    llm_sidebar()

provider = st.session_state.get("active_provider", st.session_state.provider_choice)
model = st.session_state.get(
    "active_model",
    st.session_state.openai_model_choice
    if provider == "openai"
    else st.session_state.ollama_model,
)
api_key = st.session_state.get("active_api_key", st.session_state.get("openai_api_key"))

st.title(
    "NeuroMANCER-GPT",
    anchor=False,
)

has_prompt = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)


if not has_prompt and not has_message_history:
    st.session_state.messages = []
    st.markdown(
        """
<style>
[data-testid="stMain"] {
    justify-content: center !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: -12vh !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
    with st.container():
        st.chat_input("Ask a question...", key="initial_question")
    st.stop()


user_message = st.chat_input("Ask a follow-up...")

if not user_message and has_prompt:
    user_message = st.session_state.initial_question


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container() 
            st.markdown(sanitize_markdown(message["content"]))
            if message.get("caption"):
                st.caption(message["caption"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- `{source}`")
        else:
            st.markdown(message["content"])

if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            t0 = time.perf_counter()
            history = get_recent_context(n=3)
            # retrieval sees only the current message, so resolve follow-ups
            # ("how do I change the horizon?") against the conversation first
            rewrite = contextualize_query(
                user_message,
                history,
                provider=provider,
                model=model,
                api_key=api_key,
            )
            chunks = retrieve_and_rerank(rewrite.query)
            research_s = time.perf_counter() - t0

        with st.spinner("Streaming..."):
            t1 = time.perf_counter()
            with st.container():
                response = st.write_stream(
                    stream_from_chunks(
                        user_message,
                        chunks,
                        history=history,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                    )
                )
            stream_s = time.perf_counter() - t1
            response = sanitize_markdown(
                response if isinstance(response, str) else str(response)
            )
            status_caption = build_status_caption(research_s, stream_s, rewrite)
            st.caption(status_caption)
            sources = sources_from_chunks(chunks)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- `{source}`")

    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources,
            "caption": status_caption,
        }
    )

    st.session_state.initial_question = None
    st.rerun()
