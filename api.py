from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from typing import Dict, Optional
from pathlib import Path

import logging
import threading

from contextlib import asynccontextmanager

from src.config import Config
from src.embedding import Embedder
from src.chunking import Chunker
from src.vector_db import FAISSVectorDB, QdrantVectorDB  
from src.retrieval import Retriever
from src.generation import Generator

from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    filter_source: Optional[str] = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API...")
    yield
    logger.info("Shutting down RAG API...")



app = FastAPI(
    title="Personal Knowledge Base RAG API",
    description="Simple RAG system for documents (TXT/PDF)",
    version="1.0.0",
    lifespan=lifespan
)

config = Config()

embedder = Embedder(config.get_embedding_model())
chunker = Chunker(
    strategy=config.get("chunking", {}).get("strategy", "fixed"),
    chunk_size=config.get("chunking", {}).get("size", 512)
)

db_type = config.get("vector_db", {}).get("type", "faiss")
db_path = config.get("vector_db", {}).get("path", "db/index")
dim = embedder.model.get_sentence_embedding_dimension()

faiss_lock = None

if db_type == "faiss":
    vector_db = FAISSVectorDB(db_path, dim)
    faiss_lock = threading.Lock()
elif db_type == "qdrant":
    vector_db = QdrantVectorDB(db_path)
else:
    raise ValueError(f"Unsupported vector DB: {db_type}")


retriever = Retriever(
    db=vector_db,
    embedder=embedder,
    reranker_model=config.get("reranker", {}).get("model")
)
generator = Generator(
    provider=config.get("llm", {}).get("provider", "groq"),
    model=config.get("llm", {}).get("model", "llama-3.3-70b-versatile")
)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.get("/health")
async def health_check():
    return {"status": "healthy", "db_type": db_type}

@app.post("/upload", summary="Upload and ingest a single document")
async def upload_document(
    file: UploadFile = File(...),
    source_name: Optional[str] = None  
):
    """
    Upload a TXT or PDF file → chunk → embed → store in vector DB.
    """
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    try:
        content = await file.read()
        text = content.decode("utf-8") if file.filename.endswith(".txt") else ""

        if file.filename.endswith(".pdf"):
            from io import BytesIO
            from PyPDF2 import PdfReader
            reader = PdfReader(BytesIO(content))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            text = " ".join(pages_text)
            logger.info(f"Extracted text length from PDF: {len(text)}")

        if not text.strip():
            raise HTTPException(400, detail="File is empty or could not extract text")

        # Ingest single document
        chunks = chunker.chunk(text)
        if not chunks:
            raise HTTPException(400, detail="No chunks created from document")

        vectors = embedder.embed(chunks)
        metadata = [
            {"text": chunk, "source": source_name or file.filename}
            for chunk in chunks
        ]
        if db_type == "faiss":
            with faiss_lock:
                vector_db.add(vectors, metadata)
        else:
            vector_db.add(vectors, metadata)


        logger.info(f"Ingested {len(chunks)} chunks from {file.filename}")

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": len(chunks),
            "source": source_name or file.filename
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", summary="Ask a question against the knowledge base")
async def ask_question(payload: QueryRequest):
    """
    Send JSON like: {"query": "What are embeddings?", "filter_source": "optional.pdf"}
    Returns answer + contexts.
    """
    query = payload.query
    filters = {"source": payload.filter_source} if payload.filter_source else None

    try:
        contexts = retriever.retrieve(
            query=query,
            top_k_initial=10,
            top_k_final=3,
            filters=filters
        )

        if not contexts:
            return {"answer": "No relevant information found.", "contexts": []}

        answer = generator.generate(query, contexts)

        return {
            "answer": answer,
            "contexts": [
                {"text": c["text"], "source": c.get("source", "unknown")}
                for c in contexts
            ]
        }

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(500, detail=f"Query processing failed: {str(e)}")

