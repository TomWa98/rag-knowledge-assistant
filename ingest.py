from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/documents")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents():
    documents = []

    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        documents.append({
            "source": path.name,
            "text": text
        })

    return documents


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()

        if chunk:
            chunks.append(chunk)

        if start + size >= len(text):
            break

    return chunks

def index_chunks(documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="knowledge_chunks"
    )

    for document in documents:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):
            chunk_id = f"{document['source']}_chunk_{index}"

            embedding = model.encode(chunk).tolist()

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": document["source"],
                    "chunk_index": index
                }]
            )

            print(f"Indexed: {chunk_id}")

def main():
    documents = load_documents()

    for document in documents:
        chunks = chunk_text(document["text"])

        print(f"\nDocument: {document['source']}")
        print(f"Number of chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks):
            print(f"\n--- Chunk {index} ---")
            print(chunk[:200], "...")
    
    index_chunks(documents)


if __name__ == "__main__":
    main()