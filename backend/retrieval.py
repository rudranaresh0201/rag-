from __future__ import annotations

import re
from typing import Any, List, TypedDict

from sentence_transformers import SentenceTransformer

try:
    from .db import get_collection
except ImportError:
    from db import get_collection

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does",
    "for", "from", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "please", "show", "tell", "that", "the", "their", "them", "there", "these", "they",
    "this", "to", "us", "was", "we", "what", "when", "where", "which", "who", "why", "with", "would",
    "you", "your", "explain", "describe", "give", "about",
}


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


def _rerank_score(semantic_score: float, chunk_text: str, query_terms: set[str]) -> float:
    if not query_terms:
        return semantic_score

    chunk_tokens = _tokenize(chunk_text)
    overlap_terms = query_terms & chunk_tokens
    overlap_count = float(len(overlap_terms))
    overlap_ratio = overlap_count / max(1.0, float(len(query_terms)))

    return semantic_score + (0.45 * overlap_count) + (0.8 * overlap_ratio)


def _is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return False

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return False

    similarity = float(intersection) / float(union)
    return similarity >= threshold


def _normalize_query_text(query: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(query).lower())
    tokens = [token for token in cleaned.split() if token and token not in STOPWORDS]
    return " ".join(tokens)


def _build_query_variations(query: str) -> list[str]:
    original = " ".join(str(query).split()).strip()
    simplified = _normalize_query_text(original)
    simplified_tokens = [token for token in simplified.split() if token]

    # Keyword-only variation keeps compact high-signal tokens.
    keyword_tokens = sorted(set(simplified_tokens), key=lambda token: (-len(token), token))[:6]
    keyword_only = " ".join(keyword_tokens)

    variations: list[str] = []
    for candidate in [original, simplified, keyword_only]:
        normalized = " ".join(candidate.split()).strip()
        if normalized and normalized not in variations:
            variations.append(normalized)

    return variations or [original]


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
    print("[RAG] Retrieval started")
    collection = get_collection()
    print("TOTAL CHUNKS IN DB:", collection.count())
    print("QUERY:", query)
    query_variations = _build_query_variations(query)
    print("[RAG] Query variations:", query_variations)
    query_terms = _tokenize(_normalize_query_text(query))
    query_terms_ordered = [
        token for token in _normalize_query_text(query).split() if token
    ]

    where = {"doc_id": document_id} if document_id else None
    query_results: list[dict[str, Any]] = []
    n_results = max(top_k, 6)
    embed_model = _get_embed_model()

    for variation in query_variations:
        variation_embedding = embed_model.encode(variation, show_progress_bar=False).tolist()
        results = collection.query(
            query_embeddings=[variation_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs_for_variation = results.get("documents", []) or []
        variation_chunk_count = sum(len(doc_list) for doc_list in docs_for_variation if isinstance(doc_list, list))
        print(f"[RAG] Variation '{variation}' retrieved {variation_chunk_count} chunks")
        query_results.append(results)

    candidates: list[tuple[float, RetrievedChunk]] = []

    for results in query_results:
        docs = results.get("documents", []) or []
        metadatas = results.get("metadatas", []) or []
        distances = results.get("distances", []) or []

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

    scored_chunks: list[tuple[float, RetrievedChunk]] = []
    for semantic_score, chunk in candidates:
        combined_score = _rerank_score(semantic_score, chunk["text"], query_terms)
        scored_chunks.append((combined_score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    selected_chunks: list[RetrievedChunk] = []
    seen_texts: set[str] = set()
    max_selected = min(4, max(3, top_k))
    for score, chunk in scored_chunks:
        text_key = _normalize_chunk_text(chunk["text"]).lower()
        if not text_key or text_key in seen_texts:
            continue

        duplicate_like = False
        for existing in selected_chunks:
            if _is_near_duplicate(text_key, _normalize_chunk_text(existing["text"]).lower()):
                duplicate_like = True
                break
        if duplicate_like:
            continue

        seen_texts.add(text_key)
        selected_chunks.append(chunk)
        if len(selected_chunks) >= max_selected:
            break

    if "attention" in query_terms:
        has_attention = any("attention" in _normalize_chunk_text(c["text"]).lower() for c in selected_chunks)
        if not has_attention:
            for _, chunk in scored_chunks:
                candidate_text = _normalize_chunk_text(chunk["text"]).lower()
                if "attention" in candidate_text:
                    replaced = False
                    for i, existing in enumerate(selected_chunks):
                        if "attention" not in _normalize_chunk_text(existing["text"]).lower():
                            selected_chunks[i] = chunk
                            replaced = True
                            break
                    if not replaced and len(selected_chunks) < max_selected:
                        selected_chunks.append(chunk)
                    break

    print(f"[RAG] Retrieved {len(selected_chunks)} chunks")
    for idx, chunk in enumerate(selected_chunks, start=1):
        chunk_text = _normalize_chunk_text(chunk.get("text", ""))
        preview = chunk_text[:150]
        present_terms = [term for term in query_terms_ordered if term in chunk_text.lower()]
        print(f"[RAG] Final chunk {idx}: {preview}")
        print(f"[RAG] Chunk {idx} query keywords: {present_terms}")

    context_chunks: List[str] = []
    for i, chunk in enumerate(selected_chunks):
        doc = chunk.get("file") or "Uploaded Doc"
        page = chunk.get("page")
        source_label = f"{doc} (Page {page})" if page is not None else str(doc)
        excerpt = _normalize_chunk_text(chunk["text"])[:1200]
        context_chunks.append(f"[Source: {source_label}]\n{excerpt}")

    context = "\n\n".join(context_chunks)
    return {"chunks": selected_chunks, "context": context}
