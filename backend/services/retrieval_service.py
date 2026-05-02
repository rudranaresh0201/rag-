from __future__ import annotations

from retrieval import retrieve_chunks
from llm_router import generate_answer


def retrieve_chunks_sync(query: str, top_k: int, document_id: str | None) -> dict:
    return retrieve_chunks(query=query, top_k=top_k, document_id=document_id)


def generate_answer_sync(query: str, context: str) -> str:
    return generate_answer(query, context)
