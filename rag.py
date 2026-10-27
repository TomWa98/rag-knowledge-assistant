
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# Ressourcen einmalig initialisieren
model = SentenceTransformer("all-MiniLM-L6-v2")

db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_collection(name="knowledge_chunks")

llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def retrieve(question, n_results=2):
    """Find relevant document chunks for a question."""

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    chunks = []

    for index, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]

        chunks.append({
            "text": text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": results["distances"][0][index]
        })

    return chunks


def ask_rag(question):
    """Retrieve context and generate an answer."""

    chunks = retrieve(question)

    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"[{chunk['source']}, chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a knowledge assistant.

Answer using only the provided context.
If the answer is not in the context, say:
"I cannot answer this based on the provided documents."

Do not invent facts or use external knowledge.
Cite the source filename and chunk index.
"""

    response = llm_client.chat.completions.create(
        model="qwen3:1.7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}\n/no_think"
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    return answer, chunks


if __name__ == "__main__":
    question = input("Ask a question: ")

    answer, chunks = ask_rag(question)

    print("\n--- Answer ---")
    print(answer)

    print("\n--- Retrieved Sources ---")

    for chunk in chunks:
        print(
            f"{chunk['source']} | "
            f"Chunk {chunk['chunk_index']} | "
            f"Distance: {chunk['distance']:.4f}"
        )
