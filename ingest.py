"""
INGESTION PIPELINE (run this once, and again any time you add/update documents)

This file does Steps 1, 2, and 3 of RAG:
  1. Load documents from the data/ folder
  2. Chunk them into smaller pieces
  3. Convert each chunk into an embedding and store it in a vector database (ChromaDB)

Run it with: python ingest.py
"""

import os
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DATA_FOLDER = "data"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "my_documents"
CHUNK_SIZE = 500      # number of characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks, so we don't cut an idea in half


def load_pdf_text(filepath):
    """Reads a PDF file and returns all its text as one big string."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_documents(folder=DATA_FOLDER):
    """
    Loads every .pdf and .txt file from the data folder.
    Returns a list of dicts: {"filename": ..., "text": ...}
    """
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filename.lower().endswith(".pdf"):
            text = load_pdf_text(filepath)
            documents.append({"filename": filename, "text": text})
        elif filename.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"filename": filename, "text": text})
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits a long piece of text into smaller, overlapping chunks.
    The overlap helps avoid cutting an important sentence exactly in half
    at a chunk boundary.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but keep some overlap
    return chunks


def build_vector_store():
    """
    Main function: loads documents, chunks them, embeds them,
    and stores everything in a local ChromaDB vector database on disk.
    """
    print("Loading embedding model... (first run downloads it, ~80MB, then it's cached)")
    embedder = DefaultEmbeddingFunction()
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete old collection if it exists, so re-running this script gives a fresh index
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    documents = load_documents()
    print(f"Found {len(documents)} document(s) in '{DATA_FOLDER}' folder.")

    if len(documents) == 0:
        print(f"No documents found! Add some .pdf or .txt files to the '{DATA_FOLDER}' folder and run this again.")
        return

    chunk_id = 0
    for doc in documents:
        chunks = chunk_text(doc["text"])
        print(f"  '{doc['filename']}' -> split into {len(chunks)} chunks")

        for chunk in chunks:
            if chunk.strip() == "":
                continue
            collection.add(
            ids=[str(chunk_id)],
            documents=[chunk],
            metadatas=[{"source": doc["filename"]}],
        )
            chunk_id += 1

    print(f"\nDone! Indexed {chunk_id} chunks into the vector database at '{CHROMA_DB_PATH}/'.")


if __name__ == "__main__":
    build_vector_store()
