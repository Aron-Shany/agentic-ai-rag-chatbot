"""FastAPI interface for the RAG chatbot."""
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag_graph import rag_graph

app = FastAPI(title="Agentic AI eBook RAG Chatbot")


class AskRequest(BaseModel):
    question: str


class Chunk(BaseModel):
    text: str
    page: Optional[int] = None
    score: float


class AskResponse(BaseModel):
    answer: str
    context_chunks: List[Chunk]
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    result = rag_graph.invoke({"question": question})
    return AskResponse(
        answer=result["answer"],
        context_chunks=result["context_chunks"],
        confidence=result["confidence"],
    )
