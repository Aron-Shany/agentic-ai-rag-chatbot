"""Temporary local RAG workflow for demo/testing without API credits."""

from typing import List, TypedDict
import re

from langchain_community.document_loaders import PyPDFLoader

from app import config


REFUSAL = (
    "I couldn't find enough information about that in the provided "
    "Agentic AI eBook."
)


class RAGState(TypedDict):
    question: str
    context_chunks: List[dict]
    answer: str
    confidence: float


_pdf_pages = None


def _load_pdf():
    global _pdf_pages

    if _pdf_pages is None:
        _pdf_pages = PyPDFLoader(config.PDF_PATH).load()

    return _pdf_pages


def _score_text(question: str, text: str) -> float:
    """Simple local keyword relevance score."""

    question_words = set(
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", question)
        if len(word) > 2
    )

    text_words = set(
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", text)
        if len(word) > 2
    )

    if not question_words:
        return 0.0

    matches = question_words.intersection(text_words)

    return len(matches) / len(question_words)


def retrieve(state: RAGState) -> dict:
    pages = _load_pdf()

    scored_pages = []

    for page in pages:
        text = page.page_content.strip()

        if not text:
            continue

        score = _score_text(state["question"], text)

        scored_pages.append(
            {
                "text": text,
                "page": page.metadata.get("page", 0),
                "score": round(score, 4),
            }
        )

    scored_pages.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "context_chunks": scored_pages[: config.TOP_K]
    }


def generate(state: RAGState) -> dict:
    chunks = state["context_chunks"]

    if not chunks:
        return {
            "answer": REFUSAL,
            "confidence": 0.0,
        }

    best_chunk = chunks[0]

    if best_chunk["score"] < config.MIN_RELEVANCE:
        return {
            "answer": REFUSAL,
            "confidence": 0.0,
        }

    # Temporary extractive answer.
    answer = best_chunk["text"]

    confidence = min(
        1.0,
        round(best_chunk["score"], 2)
    )

    return {
        "answer": answer,
        "confidence": confidence,
    }


class LocalRAGGraph:

    def invoke(self, state: dict) -> dict:
        retrieved = retrieve(state)

        state = {
            **state,
            **retrieved,
        }

        generated = generate(state)

        return {
            **state,
            **generated,
        }


rag_graph = LocalRAGGraph()