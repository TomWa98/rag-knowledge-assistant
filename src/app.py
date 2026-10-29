import sys
from pathlib import Path

# "streamlit run src/app.py" legt nur src/ in den sys.path, nicht den Projektordner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.rag import ask_rag, EXHAUSTIVE, TARGETED


# Seiteneinstellungen
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("RAG Knowledge Assistant")
st.caption("Local RAG with ChromaDB, Sentence Transformers and Ollama")

MODE_LABELS = {
    "Targeted Search": TARGETED,
    "Exhaustive Search": EXHAUSTIVE
}

mode_label = st.sidebar.radio(
    "Retrieval mode",
    list(MODE_LABELS),
    help=(
        "Targeted: searches the products named in the question, "
        "otherwise all documents together. "
        "Exhaustive: evaluates every document individually (slower)."
    )
)


def show_sources(chunks):
    with st.expander("Retrieved Sources"):
        for chunk in chunks:
            st.write(
                f"**{chunk['source']}** | "
                f"Page {chunk['page']} | "
                f"Chunk {chunk['chunk_index']} | "
                f"Distance: {chunk['distance']:.4f}"
            )
            st.text(chunk["text"])


# Chatverlauf initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []


# Bisherige Nachrichten anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "mode" in message:
            st.caption(f"Retrieval mode: {message['mode']}")

        if "sources" in message:
            show_sources(message["sources"])


# Neue Nutzerfrage entgegennehmen
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            answer, chunks = ask_rag(question, mode=MODE_LABELS[mode_label])

        st.markdown(answer)
        st.caption(f"Retrieval mode: {mode_label}")
        show_sources(chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "mode": mode_label,
        "sources": chunks
    })
