# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) project that answers questions grounded in your own documents, using Claude (Anthropic) for generation, ChromaDB as the vector database, and a local embedding model (free, no API key needed for embeddings).

## How it works (matches what we discussed)

1. **ingest.py** — loads documents from `data/`, splits them into chunks, converts each chunk into an embedding, and stores them in a local ChromaDB vector database.
2. **rag_chat.py** — when you ask a question, it embeds your question, retrieves the most relevant chunks from ChromaDB, and sends your question + those chunks to Claude to generate a grounded answer.
3. **app.py** — a simple Streamlit UI to ask questions and see the answer along with which source file it came from.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get your Anthropic API key
- Go to https://console.anthropic.com/ and create an API key.

### 3. Add your API key
Copy `.env.example` to `.env` and paste your key in:
```bash
cp .env.example .env
```
Then edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
```

### 4. Add your documents
Put any `.pdf` or `.txt` files into the `data/` folder. A sample leave policy `.txt` file is already included so you can test immediately.

### 5. Build the vector database (run this once, and again whenever documents change)
```bash
python ingest.py
```

### 6. Run the app
```bash
streamlit run app.py
```

It will open in your browser. Try asking: *"How many days of annual leave do employees get?"* or *"What is the maternity leave policy?"*

## Alternative: FastAPI instead of Streamlit

`main.py` is a FastAPI version of the same app — it reuses `rag_chat.py` exactly as is, only the interface layer is different. This is a good thing to mention in an interview: the RAG logic is decoupled from the UI, so swapping interfaces doesn't require touching the core pipeline.

```bash
pip install -r requirements-fastapi.txt
uvicorn main:app --reload
```

- Open **http://127.0.0.1:8000** for the simple web page.
- Open **http://127.0.0.1:8000/docs** for FastAPI's auto-generated interactive API docs — you can test the `/ask` endpoint directly there, no UI needed.
- You can also call it from the terminal: `curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"How many days of leave do I get?\"}"`

## Things you can improve for your resume (and talk about in interviews)

- **Re-ranking**: add a re-ranking step after retrieval to reorder chunks by relevance before sending to Claude.
- **Hybrid search**: combine this semantic (vector) search with keyword search (BM25) for better recall.
- **Chunking strategy**: try sentence-aware or paragraph-aware chunking instead of fixed character counts.
- **Evaluation**: add a small eval script that checks faithfulness — does the answer actually match the retrieved context?
- **Incremental re-indexing**: instead of deleting and rebuilding the whole collection every time, only update changed documents.
- **Deployment**: deploy the Streamlit app on Streamlit Community Cloud or Render so it's a live, working link you can share in interviews.

## Tech stack
- **LLM**: Claude (claude-sonnet-4-6) via Anthropic API
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`) — free, runs locally
- **Vector database**: ChromaDB (free, open-source)
- **UI**: Streamlit (`app.py`) or FastAPI + plain HTML/JS (`main.py`) — pick whichever you want to demo
