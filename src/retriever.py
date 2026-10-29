import json
import logging
from pathlib import Path

# Pfade relativ zum Projektordner, unabhängig vom Arbeitsverzeichnis
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "products.json"
CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "knowledge_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)

# Schwere Ressourcen erst bei Bedarf laden (schnelle Imports, testbar ohne DB)
_model = None
_collection = None


def get_embedding_model():
    """Return the shared sentence-transformer model, loading it on first use."""
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def get_collection():
    """Return the persistent ChromaDB collection, opening it on first use."""
    global _collection

    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)

    return _collection


def embed_query(question):
    return get_embedding_model().encode(question).tolist()


def load_product_catalog(path=CATALOG_PATH):
    """Load the product catalog from disk. Returns an empty list if missing."""
    path = Path(path)

    if not path.exists():
        logger.warning("Product catalog not found: %s", path)
        return []

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


PRODUCTS = load_product_catalog()


def detect_products(question, products=None):
    """Return all products explicitly mentioned in a question."""
    products = PRODUCTS if products is None else products
    question_lower = question.lower()

    return [
        product
        for product in products
        if product["model"].lower() in question_lower
    ]


def select_products(selection=None, products=None):
    """Restrict the catalog to the selected models or source filenames.

    None selects every product; unknown entries are logged and ignored.
    """
    products = PRODUCTS if products is None else products

    if selection is None:
        return list(products)

    wanted = {item.lower() for item in selection}
    selected = [
        product
        for product in products
        if product["model"].lower() in wanted
        or product["source"].lower() in wanted
    ]

    known = {p["model"].lower() for p in selected} | {p["source"].lower() for p in selected}
    for item in sorted(wanted - known):
        logger.warning("Selected document not in catalog: %s", item)

    return selected


def _results_to_chunks(results):
    """Convert a ChromaDB query result into a flat list of chunk dicts."""
    if not results:
        return []

    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []

    chunks = []

    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}

        chunks.append({
            "text": text,
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": distances[index] if index < len(distances) else float("nan")
        })

    return chunks


def deduplicate_chunks(chunks):
    """Drop repeated chunks (same source, page and chunk index), keeping order."""
    seen = set()
    unique = []

    for chunk in chunks:
        key = (chunk["source"], chunk["page"], chunk["chunk_index"])

        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return unique


def query_chunks(query_embedding, n_results=5, source=None, collection=None):
    """Run one similarity query, optionally restricted to a single source file."""
    collection = collection or get_collection()

    if collection.count() == 0:
        return []

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }

    if source is not None:
        kwargs["where"] = {"source": source}

    return _results_to_chunks(collection.query(**kwargs))


def retrieve_from_documents(query_embedding, sources, n_results=5, collection=None):
    """Search each given source file separately and return the combined chunks."""
    chunks = []

    for source in sources:
        chunks.extend(query_chunks(query_embedding, n_results, source, collection))

    return deduplicate_chunks(chunks)


def retrieve_targeted(question, n_results=5, products=None, collection=None,
                      query_embedding=None):
    """Retrieve chunks for explicitly mentioned products, else search globally.

    Each mentioned product is searched separately so every one is represented.
    """
    if query_embedding is None:
        query_embedding = embed_query(question)

    matched_products = detect_products(question, products)

    if matched_products:
        sources = [product["source"] for product in matched_products]
        return retrieve_from_documents(query_embedding, sources, n_results, collection)

    return deduplicate_chunks(
        query_chunks(query_embedding, n_results, collection=collection)
    )


def get_document_overview(source, collection=None):
    """Return the opening chunk (page 1, chunk 0) of a document, or "".

    Datasheets usually name the product there, which helps identify what kind
    of product a document describes even if that chunk ranks low.
    """
    collection = collection or get_collection()

    if collection.count() == 0:
        return ""

    result = collection.get(
        where={"$and": [{"source": source}, {"page": 1}, {"chunk_index": 0}]},
        include=["documents"]
    )
    documents = (result or {}).get("documents") or []

    return documents[0] if documents else ""


def retrieve_exhaustive(question, selection=None, n_results=5, products=None,
                        collection=None, query_embedding=None):
    """Retrieve the top chunks from every selected document individually.

    Every selected document is examined and returned, even if it yields no
    chunks. Within a document only the top `n_results` chunks are returned,
    so relevant passages ranked lower can still be missed.

    Returns a list of {"model", "source", "overview", "chunks"} dicts in
    catalog order.
    """
    selected = select_products(selection, products)

    if not selected:
        return []

    if query_embedding is None:
        query_embedding = embed_query(question)

    return [
        {
            "model": product["model"],
            "source": product["source"],
            "overview": get_document_overview(product["source"], collection),
            "chunks": deduplicate_chunks(
                query_chunks(query_embedding, n_results, product["source"], collection)
            )
        }
        for product in selected
    ]


def flatten_document_results(document_results):
    """Flatten per-document exhaustive results into one chunk list."""
    return deduplicate_chunks(
        chunk
        for document in document_results
        for chunk in document["chunks"]
    )
