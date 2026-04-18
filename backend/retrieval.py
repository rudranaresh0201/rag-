from __future__ import annotations

import re
from typing import Any, List, TypedDict

from sentence_transformers import SentenceTransformer

from db import get_collection

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
IMPORTANT_KEYWORDS = {"formula", "definition", "equation", "value", "law", "theorem", "rate"}
_embed_model: SentenceTransformer | None = None


class RetrievedChunk(TypedDict):
    text: str
    file: str
    doc_id: str
    page: int | None
    metadata: dict[str, Any]


class RetrievalResult(TypedDict):
    chunks: List[RetrievedChunk]
    context: str


def _normalize_chunk_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _clean_broken_sentences(text: str) -> str:
    normalized = _normalize_chunk_text(text)
    if not normalized:
        return ""

    segments = re.split(r"(?<=[.!?])\s+", normalized)
    cleaned_segments: list[str] = []
    for segment in segments:
        candidate = segment.strip()
        if not candidate:
            continue

        words = re.findall(r"[a-zA-Z0-9=()+\-/*^.]+", candidate)
        if len(words) < 5 and not any(token in candidate for token in ["=", "defined as", "is given by"]):
            continue

        cleaned_segments.append(candidate)

    return " ".join(cleaned_segments) if cleaned_segments else normalized


def _word_windows(text: str, min_words: int = 300, max_words: int = 500, overlap_words: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    if len(words) <= max_words:
        return [" ".join(words)]

    step = max(1, max_words - overlap_words)
    windows: list[str] = []
    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) < min_words and start > 0:
            break
        windows.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return windows or [" ".join(words)]


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2}


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> RetrievalResult:
    query_embedding = _get_embed_model().encode(query, show_progress_bar=False).tolist()
    collection = get_collection()
    print("TOTAL CHUNKS IN DB:", collection.count())
    print("QUERY:", query)

    where = {"doc_id": document_id} if document_id else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, 5),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    print("RESULT IDS:", results.get("ids"))
    print("RESULT DOCS:", results.get("documents"))

    docs = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []
    distances = results.get("distances", []) or []

    candidates: list[tuple[float, RetrievedChunk]] = []

    for i, doc_list in enumerate(docs):
        if not isinstance(doc_list, list):
            continue

        metadata_list = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], list) else []
        distance_list = distances[i] if i < len(distances) and isinstance(distances[i], list) else []

        for j, chunk in enumerate(doc_list):
            raw_text = str(chunk) if chunk else ""
            cleaned_text = _clean_broken_sentences(raw_text)
            if not cleaned_text or len(cleaned_text) <= 40:
                continue

            cleaned_text = cleaned_text.strip().replace("\n", " ")
            sub_chunks = _word_windows(cleaned_text, min_words=300, max_words=500, overlap_words=50)

            metadata = metadata_list[j] if j < len(metadata_list) and isinstance(metadata_list[j], dict) else {}

            distance_val = distance_list[j] if j < len(distance_list) else None
            semantic_score = 0.0
            if isinstance(distance_val, (int, float)):
                semantic_score = 1.0 / (1.0 + float(distance_val))

            page_val = metadata.get("page")
            page = page_val if isinstance(page_val, int) else int(page_val) if isinstance(page_val, str) and page_val.isdigit() else None

            for sub_chunk in sub_chunks:
                candidates.append(
                    (
                        semantic_score,
                        {
                            "text": sub_chunk,
                            "file": str(metadata.get("file", "unknown.pdf")),
                            "doc_id": str(metadata.get("doc_id", "")),
                            "page": page,
                            "metadata": metadata,
                        },
                    )
                )

    query_terms = _tokenize(query)
    important_query_terms = query_terms & IMPORTANT_KEYWORDS

    scored_chunks: list[tuple[float, RetrievedChunk]] = []
    for semantic_score, chunk in candidates:
        chunk_terms = _tokenize(chunk["text"])
        keyword_overlap_bonus = float(len(important_query_terms & chunk_terms))
        general_overlap_bonus = 0.2 * float(len(query_terms & chunk_terms))
        combined_score = semantic_score + keyword_overlap_bonus + general_overlap_bonus
        scored_chunks.append((combined_score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    selected_chunks: list[RetrievedChunk] = []
    seen_texts: set[str] = set()
    for _, chunk in scored_chunks:
        text_key = _normalize_chunk_text(chunk["text"]).lower()
        if not text_key or text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        selected_chunks.append(chunk)
        if len(selected_chunks) >= 3:
            break

    context_chunks: List[str] = []
    for i, chunk in enumerate(selected_chunks):
        context_chunks.append(f"[SOURCE {i + 1}] {chunk['text']}")

    context = "\n\n".join(context_chunks[:3])
    print("RETRIEVED CONTEXT:")
    print(context)
    return {"chunks": selected_chunks, "context": context}
