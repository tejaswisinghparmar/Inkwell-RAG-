"""
ingest.py — PDF Ingestion & Indexing Pipeline



Reads a PDF document, splits it into overlapping chunks, generates
vector embeddings with FastEmbed (BAAI/bge-small-en-v1.5), and stores
them in a Qdrant collection for later retrieval.

Usage:
    python rag/index.py

Prerequisites:
    - Qdrant running locally (docker-compose.yml) OR Qdrant Cloud URL + API key in .env
    - PDF placed in the same directory as this script
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────
PDF_FILENAME = "harry-potter-and-the-half-blood-prince-j.k.-rowling.pdf"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "harry_potter_fast")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
CHUNK_SIZE = 1000    # tokens per chunk
CHUNK_OVERLAP = 400  # overlap between consecutive chunks
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, fast & lightweight


def main() -> None:
    # 1. Load PDF
    pdf_path = Path(__file__).parent / PDF_FILENAME
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    print(f"📄 Loading PDF: {pdf_path.name}")
    docs = PyPDFLoader(file_path=str(pdf_path)).load()
    print(f"   → {len(docs)} pages loaded")

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents=docs)
    print(f"✂️  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # 3. Embed & store in Qdrant
    embedding_model = FastEmbedEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    kwargs = {
        "documents": chunks,
        "embedding": embedding_model,
        "url": QDRANT_URL,
        "collection_name": COLLECTION_NAME,
    }
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY

    QdrantVectorStore.from_documents(**kwargs)
    print(f"✅ Indexing complete — collection '{COLLECTION_NAME}' ready at {QDRANT_URL}")


if __name__ == "__main__":
    main()