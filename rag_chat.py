"""
RETRIEVAL + GENERATION PIPELINE (this runs every time a user asks a question)

This file does Step 4 of RAG:
  - Embed the user's question
  - Retrieve the most relevant chunks from ChromaDB (this is the vector database we built in ingest.py)
  - Send the question + retrieved chunks to Gemini
  - Return Gemini's answer, grounded in the retrieved context
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from a .env file in this folder

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "my_documents"
TOP_K = 3                      # how many chunks to retrieve per question
GEMINI_MODEL = "gemini-3.5-flash"

# Load the embedding model and connect to the vector DB once at startup,
# instead of reloading them every single time a question is asked (this matters for latency!)
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_collection = _client.get_collection(COLLECTION_NAME)
_gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def retrieve_chunks(query, top_k=TOP_K):
    """
    Converts the query into an embedding, then searches ChromaDB
    for the most semantically similar chunks (this is the "Retrieval" in RAG).
    """
    query_embedding = _embedder.encode(query).tolist()
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return chunks, sources


def generate_answer(query, chunks):
    """
    Builds a prompt that includes the retrieved chunks as context,
    and asks Gemini to answer using ONLY that context (this is "Augmentation" + "Generation").
    """
    context_text = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information to answer that."

Context:
{context_text}

Question: {query}

Answer:"""

    response = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def ask(query):
    """
    The full RAG pipeline in one function call: retrieve -> augment -> generate.
    Returns the answer plus which source files were used (useful for showing
    the user exactly where the answer came from).
    """
    chunks, sources = retrieve_chunks(query)
    answer = generate_answer(query, chunks)
    return answer, list(set(sources))


if __name__ == "__main__":
    # Quick command-line test, without the Streamlit UI
    question = input("Ask a question: ")
    answer, sources = ask(question)
    print("\nAnswer:", answer)
    print("Sources used:", sources)