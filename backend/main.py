import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm_client import ask_llm
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.retrieve import retrieve

app = FastAPI(title =" rag-core backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],

    
)
EMBEDDINGS_PATH =Path(__file__).parent.parent /"embeddings.json"

with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
    EMBEDDINGS =json.load(f)

class QueryRequest(BaseModel):
    question: str

class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_doc: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[Chunk]

def build_prompt(question: str, chunks: list)-> str:
    context="\n\n".join(
        f"[Source: {c['source_doc']}]\n{c['text']}" for c in chunks
    )
    return f""" Answer the question using ONLY the context below.
    If the context doesn't contain the answer, say " i don't have enough information to answer that" - do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

@app.get("/")
def health_check():
    return{"status": "ok", "message": "rage-core backend is running"}

@app.post("/query",response_model=QueryResponse)
def query(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="queation cannot be empty")
    start = time.time()

    try:
        top_chunks= retrieve(question, EMBEDDINGS, top_k=5)
    except Exception as e :
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")

    if not top_chunks:
        raise HTTPException(
            status_code=404,
            detail="no relevant chunks found in this question"
        )

    prompt = build_prompt(question, top_chunks)
    try:
        answer=ask_llm(prompt)

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    elapsed= round(time.time() - start, 2)
    print(f"[query] '{question[:50]}...' -> {elapsed}s, {len(top_chunks)} chunks")

    return QueryResponse(answer=answer, retrieved_chunks=top_chunks)
