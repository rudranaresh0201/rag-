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
CONCEPT_KEYWORDS = {"system", "works", "mechanism", "function", "flow", "control"}
NOISE_MARKERS = {"page", "figure", "table", "rev"}
RULEBOOK_TERMS = {
    "shall", "must", "required", "requirements", "rule", "rules", "compliance", "inspection", "team", "teams",
}
TEACHING_TERMS = {"explains", "means", "works", "because", "therefore", "allows", "used", "process", "mechanism"}
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

    conceptual_hits = sum(1 for keyword in CONCEPT_KEYWORDS if keyword in chunk_tokens)
    conceptual_boost = 0.22 * float(conceptual_hits)

    rulebook_hits = sum(1 for keyword in RULEBOOK_TERMS if keyword in chunk_tokens)
    rulebook_penalty = 0.22 * float(rulebook_hits)

    return semantic_score + (0.45 * overlap_count) + (0.8 * overlap_ratio) + conceptual_boost - rulebook_penalty


def _digit_ratio(text: str) -> float:
    alnum_chars = [ch for ch in text if ch.isalnum()]
    if not alnum_chars:
        return 0.0
    digit_chars = [ch for ch in alnum_chars if ch.isdigit()]
    return float(len(digit_chars)) / float(len(alnum_chars))


def _contains_rule_reference(text: str) -> bool:
    return bool(re.search(r"\b[A-Za-z]\.\d+(?:\.\d+)+\b", text))


def _contains_noise_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    # Include variants like rev01 in addition to exact words.
    return any(marker in lowered for marker in NOISE_MARKERS)


def _quality_signals(text: str) -> tuple[int, int, int]:
    tokens = _tokenize(text)
    conceptual_hits = sum(1 for keyword in CONCEPT_KEYWORDS if keyword in tokens)
    rulebook_hits = sum(1 for keyword in RULEBOOK_TERMS if keyword in tokens)
    teaching_hits = sum(1 for keyword in TEACHING_TERMS if keyword in tokens)
    return conceptual_hits, rulebook_hits, teaching_hits


def _is_preferred_chunk(text: str) -> bool:
    conceptual_hits, rulebook_hits, teaching_hits = _quality_signals(text)
    has_heavy_rulebook = rulebook_hits >= 2
    has_teaching_signal = teaching_hits >= 1 or conceptual_hits >= 2
    return (not has_heavy_rulebook) and has_teaching_signal


def _strip_noisy_lines(text: str) -> tuple[str, float]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return "", 1.0

    # Many chunks arrive as a single long line; split into sentence-like spans first.
    spans = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", raw_text) if segment.strip()]
    if not spans:
        spans = [raw_text.strip()]

    clean_lines: list[str] = []
    noisy_count = 0

    for line in spans:
        noisy_marker = _contains_noise_marker(line)
        noisy_rule = _contains_rule_reference(line)
        noisy_digits = _digit_ratio(line) > 0.30

        if noisy_marker or noisy_rule or noisy_digits:
            noisy_count += 1

            # Try to salvage conceptual text by removing rule/spec artifacts.
            sanitized = re.sub(r"\b[A-Za-z]\.\d+(?:\.\d+)+\b", " ", line)
            sanitized = re.sub(r"\b(Page|Figure|Table|Rev)\b", " ", sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r"\s+", " ", sanitized).strip()

            alpha_chars = [ch for ch in sanitized if ch.isalpha()]
            if sanitized and len(sanitized) >= 45 and len(alpha_chars) >= 25:
                clean_lines.append(sanitized)
            continue

        clean_lines.append(line)

    noisy_ratio = float(noisy_count) / float(max(1, len(spans)))
    return "\n".join(clean_lines).strip(), noisy_ratio


def _is_noisy_chunk(text: str) -> bool:
    candidate = str(text or "")
    if not candidate.strip():
        return True

    if _contains_rule_reference(candidate):
        return True

    # Treat numeric-heavy content as spec/rulebook noise.
    if _digit_ratio(candidate) > 0.30:
        return True

    return False


def _clip_excerpt(text: str, max_chars: int = 480) -> str:
    normalized = _normalize_chunk_text(text)
    if len(normalized) <= max_chars:
        return normalized

    clipped = normalized[:max_chars]
    last_break = max(clipped.rfind("."), clipped.rfind(";"), clipped.rfind(":"))
    if last_break > int(max_chars * 0.6):
        clipped = clipped[: last_break + 1]
    return clipped.strip()


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
    print(f"[RAG] Retrieval using collection: {collection.name}")
    print("[RAG] Total chunks in DB:", collection.count())
    query_variations = _build_query_variations(query)
    print("[RAG] Query variations:", query_variations)
    query_terms = _tokenize(_normalize_query_text(query))
    query_terms_ordered = [
        token for token in _normalize_query_text(query).split() if token
    ]

    where = {"doc_id": document_id} if document_id else None
    query_results: list[dict[str, Any]] = []
    # Keep retrieval broad, but enforce strict final context selection later.
    n_results = max(top_k, 8)
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
    filter_stats = {
        "dropped_noisy_ratio": 0,
        "dropped_too_short": 0,
        "dropped_noisy_chunk": 0,
        "dropped_post_clip_noisy": 0,
    }

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
                denoised_text, noisy_ratio = _strip_noisy_lines(raw_text)
                if noisy_ratio >= 0.95:
                    filter_stats["dropped_noisy_ratio"] += 1
                    continue

                cleaned_text = _clean_broken_sentences(denoised_text)
                if not cleaned_text or len(cleaned_text) <= 40:
                    filter_stats["dropped_too_short"] += 1
                    continue

                if _is_noisy_chunk(cleaned_text):
                    filter_stats["dropped_noisy_chunk"] += 1
                    continue

                cleaned_text = cleaned_text.strip().replace("\n", " ")
                clipped_text = _clip_excerpt(cleaned_text, max_chars=480)
                if not clipped_text or _is_noisy_chunk(clipped_text):
                    filter_stats["dropped_post_clip_noisy"] += 1
                    continue

                metadata = metadata_list[j] if j < len(metadata_list) and isinstance(metadata_list[j], dict) else {}

                distance_val = distance_list[j] if j < len(distance_list) else None
                semantic_score = 0.0
                if isinstance(distance_val, (int, float)):
                    semantic_score = 1.0 / (1.0 + float(distance_val))

                page_val = metadata.get("page")
                page = page_val if isinstance(page_val, int) else int(page_val) if isinstance(page_val, str) and page_val.isdigit() else None

                candidates.append(
                    (
                        semantic_score,
                        {
                            "text": clipped_text,
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
    max_selected = 3

    preferred_scored = [item for item in scored_chunks if _is_preferred_chunk(item[1].get("text", ""))]
    fallback_scored = [item for item in scored_chunks if not _is_preferred_chunk(item[1].get("text", ""))]

    for score, chunk in preferred_scored + fallback_scored:
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
        excerpt = _clip_excerpt(chunk["text"], max_chars=480)
        context_chunks.append(f"[Source: {source_label}]\n{excerpt}")

    context = "\n\n".join(context_chunks)
    context = _normalize_chunk_text(context)
    if len(context) > 2000:
        context = context[:2000].strip()

    clean_previews = [_normalize_chunk_text(chunk.get("text", ""))[:120] for chunk in selected_chunks]
    print("[RAG] Clean chunks used:", clean_previews)
    print("[RAG] Final context length:", len(context))
    return {"chunks": selected_chunks, "context": context}
