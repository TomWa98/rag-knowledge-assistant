from pathlib import Path
import json

from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# Pfade relativ zum Projektordner, unabhängig vom Arbeitsverzeichnis
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
CATALOG_PATH = PROJECT_ROOT / "data" / "products.json"
CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")

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

def load_pdfs():
    documents = []

    for path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "source": path.name,
                    "page": page_number,
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

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(name="knowledge_chunks")
    except chromadb.errors.NotFoundError:
        pass

    collection = client.get_or_create_collection(
        name="knowledge_chunks"
    )

    for document in documents:
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):
            chunk_id = (
                f"{document['source']}_"
                f"page_{document['page']}_chunk_{index}"
            )

            embedding = model.encode(chunk).tolist()

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_index": index
                }]
            )

            print(f"Indexed: {chunk_id}")


def save_product_catalog(documents):
    """Create a product catalog from indexed PDF filenames."""

    sources = sorted({doc["source"] for doc in documents})

    products = [
        {
            "model": Path(source).stem,
            "source": source
        }
        for source in sources
    ]

    catalog_path = CATALOG_PATH
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    with catalog_path.open("w", encoding="utf-8") as file:
        json.dump(products, file, indent=2)

    print(f"Saved product catalog with {len(products)} products")


def main():
    documents = load_pdfs()

    print(f"Extracted PDF pages: {len(documents)}")

    if not documents:
        print("No PDF content found.")
        return

    index_chunks(documents)
    save_product_catalog(documents)

    print("\nPDF indexing completed.")


if __name__ == "__main__":
    main()