from __future__ import annotations

import html
import time

import streamlit as st

from generate import sources_from_chunks, stream_from_chunks
from retrieve import retrieve, warmup


CHAT_CSS = """
<style>
/* Hide default chat avatars and collapse their column */
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

/* User query: stronger color that stays readable in light and dark */
.user-query {
    color: #1d4ed8;
    font-weight: 500;
    margin: 0;
    white-space: pre-wrap;
}
@media (prefers-color-scheme: dark) {
    .user-query { color: #93c5fd; }
}
/* Streamlit theme toggles (overrides OS preference when set) */
[data-theme="dark"] .user-query,
.stApp[data-theme="dark"] .user-query,
html[data-theme="dark"] .user-query {
    color: #93c5fd !important;
}
[data-theme="light"] .user-query,
.stApp[data-theme="light"] .user-query,
html[data-theme="light"] .user-query {
    color: #1d4ed8 !important;
}
</style>
"""


@st.cache_resource
def init_rag():
    warmup()
    return True


@st.cache_data(ttl=1800)
def retrieve_and_rerank(query: str):
    return retrieve(query)


def get_recent_context(n: int = 3) -> list[dict]:
    prior = st.session_state.messages[:-1]
    return [{"role": m["role"], "content": m["content"]} for m in prior[-n:]]


def render_user_message(content: str) -> None:
    st.markdown(
        f'<p class="user-query">{html.escape(content)}</p>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="NeuroMANCER GPT", page_icon="")
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    st.title("NeuroMANCER GPT")

    init_rag()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                render_user_message(message["content"])
            else:
                st.markdown(message["content"])
                if message.get("elapsed_s") is not None:
                    st.caption(f"{message['elapsed_s']:.1f}s")
                if message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- `{source}`")

    prompt = st.chat_input("Ask a question about NeuroMANCER…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        render_user_message(prompt)

    history = get_recent_context(n=3)

    with st.chat_message("assistant"):
        try:
            t0 = time.perf_counter()
            chunks = retrieve_and_rerank(prompt)
            answer = st.write_stream(
                stream_from_chunks(prompt, chunks, history=history)
            )
            elapsed_s = time.perf_counter() - t0
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.session_state.messages.pop()
            return

        sources = sources_from_chunks(chunks)
        st.caption(f"{elapsed_s:.1f}s")
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(f"- `{source}`")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer if isinstance(answer, str) else str(answer),
            "sources": sources,
            "elapsed_s": elapsed_s,
        }
    )


if __name__ == "__main__":
    main()
