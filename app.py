
import streamlit as st
from rag import ask_rag


# Seiteneinstellungen
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("RAG Knowledge Assistant")
st.caption("Local RAG with ChromaDB, Sentence Transformers and Ollama")


# Chatverlauf initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []


# Bisherige Nachrichten anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message:
            with st.expander("Retrieved Sources"):
                for chunk in message["sources"]:
                    st.write(
                        f"**{chunk['source']}** | "
                        f"Chunk {chunk['chunk_index']} | "
                        f"Distance: {chunk['distance']:.4f}"
                    )
                    st.text(chunk["text"])


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
            answer, chunks = ask_rag(question)

        st.markdown(answer)

        with st.expander("Retrieved Sources"):
            for chunk in chunks:
                st.write(
                    f"**{chunk['source']}** | "
                    f"Chunk {chunk['chunk_index']} | "
                    f"Distance: {chunk['distance']:.4f}"
                )
                st.text(chunk["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks
    })
