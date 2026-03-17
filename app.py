"""
FastAPI application for Enterprise Knowledge AI Platform
Exposes REST endpoints for document ingestion and NL querying
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil, os, tempfile

from rag_pipeline import init_pinecone, ingest_documents, build_rag_chain
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

app = FastAPI(
    title="Enterprise Knowledge AI Platform",
    description="Natural language querying across 500K+ enterprise documents via RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "enterprise-knowledge")

# Initialize on startup
init_pinecone()
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)
rag_chain = build_rag_chain(vectorstore)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", summary="Upload and ingest documents into the knowledge base")
async def ingest(files: List[UploadFile] = File(...)):
    saved_paths = []
    try:
        for file in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
            shutil.copyfileobj(file.file, tmp)
            tmp.close()
            saved_paths.append(tmp.name)
        ingest_documents(saved_paths)
        return {"message": f"Successfully ingested {len(files)} document(s)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in saved_paths:
            os.unlink(p)


@app.post("/query", response_model=QueryResponse, summary="Run a natural language query")
def query(req: QueryRequest):
    try:
        result = rag_chain({"query": req.question})
        return QueryResponse(
            answer=result["result"],
            sources=[doc.metadata for doc in result["source_documents"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
