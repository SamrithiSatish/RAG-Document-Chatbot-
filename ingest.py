import os
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()


DOCS_DIR = "documents"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


embedder = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(collection_name):
    """Get or create a Chroma collection scoped to a specific session/user."""
    return client.get_or_create_collection(collection_name)


def load_text(filepath):
    """Load text from a local file (PDF or plain text)."""
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_and_store(text, source_name, collection):
    """Chunk, embed, and store a piece of text in the given collection."""
    if not text.strip():
        print(f"  Skipping {source_name} — no extractable text.")
        return False

    chunks = chunk_text(text)
    if not chunks:
        print(f"  Skipping {source_name} — no chunks produced.")
        return False

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return True


def ingest_documents(collection_name="documents"):
    """CLI helper: ingest every file in documents/ into a given (default: shared) collection."""
    collection = get_collection(collection_name)
    doc_count = 0
    for filename in os.listdir(DOCS_DIR):
        if filename.startswith("."):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        print(f"Processing {filename}...")
        text = load_text(filepath)
        if process_and_store(text, source_name=filename, collection=collection):
            doc_count += 1

    print(f"\nIngested {doc_count} document(s) into Chroma collection '{collection_name}'.")


if __name__ == "__main__":
    ingest_documents()