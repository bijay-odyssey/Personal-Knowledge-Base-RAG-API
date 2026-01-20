# Personal Knowledge Base RAG API

A modular Retrieval-Augmented Generation (RAG) system that allows users to upload TXT or PDF documents and query them using natural language questions. The API is built with FastAPI and uses local vector storage (FAISS or Qdrant), sentence-transformers embeddings, optional cross-encoder reranking, and Groq for fast LLM inference.

## Features

- Document ingestion endpoint (`/upload`) for single TXT or PDF files
- Question-answering endpoint (`/query`) with optional source filtering
- Switchable vector database: FAISS (in-memory/index file) or Qdrant (persistent)
- Configurable chunking strategies: fixed-size, sentence-based, or basic recursive
- Cosine similarity search with optional BGE-style reranking
- Groq API integration for low-latency generation (Llama-3.3-70B or similar models)
- Thread-safe FAISS writes using lock
- Health check endpoint
- Automatic interactive Swagger UI at `/docs`

## Project Structure

```
.
├── src/
│   ├── api.py              # FastAPI application and endpoints
│   ├── config.py           # YAML configuration loader
│   ├── embedding.py        # SentenceTransformer wrapper
│   ├── chunking.py         # Chunking logic (fixed, sentence, recursive)
│   ├── vector_db.py        # Abstract VectorDB + FAISS/Qdrant implementations
│   ├── retrieval.py        # Retrieval + optional reranking
│   └── generation.py       # Groq LLM generation
├── config.yaml             # Configuration file (models, DB type, chunk size, etc.)
├── requirements.txt        # Dependencies
├── .env                    # Environment variables (GROQ_API_KEY)
└── README.md
```

## Prerequisites

- Python 3.10+
- Groq API key (set as environment variable `GROQ_API_KEY`)
- (Optional) Docker if you plan to run Qdrant in container mode later

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/bijay-odyssey/Personal-Knowledge-Base-RAG-API.git
   cd Personal-Knowledge-Base-RAG-API
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   # or
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set your Groq API key:

   Create a `.env` file in the root:

   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   Or export it directly:

   ```bash
   export GROQ_API_KEY="gsk_..."
   ```

## Configuration (config.yaml)

Customize models, database backend, and chunking:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"

vector_db:
  type: "faiss"           # or "qdrant"
  path: "db/index"        # for FAISS; ignored for in-memory Qdrant

chunking:
  strategy: "fixed"       # "fixed", "sentence", "recursive"
  size: 512

reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile"
```

## Running the API

Start the server with hot-reload (development):

```bash
uvicorn src.api:app --reload --port 8000
```

Or in production mode:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs in your browser to access the interactive Swagger UI.

## API Endpoints

### POST /upload

Upload and index a document.

**Form-data:**
- `file`: TXT or PDF file
- `source_name` (optional): custom label for filtering

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/notes.pdf" \
  -F "source_name=week1-notes"
```

### POST /query

Ask a question against indexed documents.

**JSON body:**

```json
{
  "query": "What are embeddings?",
  "filter_source": "week1-notes"    // optional
}
```

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain embeddings", "filter_source": "notes.pdf"}'
```

### GET /health

Check API status.

```bash
curl http://localhost:8000/health
```

## Development Notes

- FAISS uses a global lock for thread-safe writes in concurrent uploads.
- Qdrant collection is recreated on startup (for development); in production, change to load existing collection.
- PDF text extraction uses PyPDF2 (simple; no layout preservation).
- Reranking is optional — set `reranker.model` to empty string to disable.
- No persistent storage cleanup — delete `db/index` or Qdrant data directory manually when resetting.

## Future Improvements

- Multi-file / folder ingestion
- Background task processing for large uploads
- Advanced chunking (overlap, semantic)
- Evaluation endpoints
- Authentication & rate limiting
- Qdrant Docker integration
- Gradio / Streamlit frontend


