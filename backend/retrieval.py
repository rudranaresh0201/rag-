from __future__ import annotations

import re
from typing import Any, List, TypedDict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .db import get_collection, get_embedder

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_embed_model: SentenceTransformer | None = None
CONCEPT_KEYWORDS = {"system", "works", "mechanism", "function", "flow", "control"}
NOISE_MARKERS = {"page", "figure", "table", "rev"}
RULEBOOK_TERMS = {
    "shall", "must", "required", "requirements", "rule", "rules", "compliance", "inspection", "team", "teams",
}
TEACHING_TERMS = {"explains", "means", "works", "because", "therefore", "allows", "used", "process", "mechanism"}
EXPLANATORY_VERBS = {"is", "are", "uses", "converts", "maps", "explains", "works", "operates"}
DEFINITION_PATTERNS = (" is a ", " refers to ", " defined as ")
INSTRUCTION_MARKERS = {"verify", "note that", "question", "name", "compare", "identify"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does",
    "for", "from", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "please", "show", "tell", "that", "the", "their", "them", "there", "these", "they",
    "this", "to", "us", "was", "we", "what", "when", "where", "which", "who", "why", "with", "would",
    "you", "your", "explain", "describe", "give", "about",
}
KEYWORD_MIN_LEN = 4
KEYWORD_VARIANTS = {
    "quantisation": "quantization",
    "colour": "color",
    "behaviour": "behavior",
}
KEYWORD_SYNONYMS = {
    "sampling": ["sampling", "sample"],
    "quantization": ["quantization", "quantisation"],
    "encoding": ["encoding", "encode"],
}
CORE_TERM_STOPWORDS = {
    "and", "or", "the", "a", "an", "what", "how", "explain", "describe", "about", "theorem",
}
NON_CONTENT_QUERY_WORDS = {
    "what", "is", "the", "explain", "story", "describe", "moral",
}
GENERIC_QUERY_WORDS = {"system", "explain", "what", "how"}
GENERIC_MATCH_WORDS = {
    "system", "control", "data", "process", "method", "model", "design", "working", "overview",
}
DOMAIN_QUERY_TRIGGERS = {"vehicle", "electric", "battery", "motor", "tractive"}
DOMAIN_BOOST_TERMS = {"ev", "baja", "vehicle", "battery", "motor"}
TITLE_NOISE_PHRASES = {
    "design and operation of a modern smart city infrastructure",
    "smart city infrastructure report",
    "technical reference document",
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


def _normalize_keyword_token(token: str) -> str:
    base = str(token or "").lower().strip()
    return KEYWORD_VARIANTS.get(base, base)


def _normalize_keyword_text(text: str) -> str:
    lowered = str(text or "").lower()
    for variant, canonical in KEYWORD_VARIANTS.items():
        lowered = re.sub(rf"\b{re.escape(variant)}\b", canonical, lowered)
    return lowered


def _expand_keywords(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        normalized = _normalize_keyword_token(token)
        if normalized and normalized not in seen:
            seen.add(normalized)
            expanded.append(normalized)

        synonym_candidates = KEYWORD_SYNONYMS.get(normalized, [normalized])
        for synonym in synonym_candidates:
            normalized_syn = _normalize_keyword_token(synonym)
            if normalized_syn and normalized_syn not in seen:
                seen.add(normalized_syn)
                expanded.append(normalized_syn)

    return expanded


def _tokenize_name(text: str) -> set[str]:
    return {
        _normalize_keyword_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())
        if len(token) >= KEYWORD_MIN_LEN and _normalize_keyword_token(token) not in STOPWORDS
    }


def _extract_query_keywords(query: str) -> list[str]:
    raw_query = str(query or "")
    base_tokens = [
        _normalize_keyword_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", raw_query.lower())
        if (
            len(token) >= 3
            and _normalize_keyword_token(token) not in STOPWORDS
            and _normalize_keyword_token(token) not in NON_CONTENT_QUERY_WORDS
        )
    ]
    return _expand_keywords(list(dict.fromkeys(base_tokens)))


def _extract_core_query_terms(query: str) -> list[str]:
    tokens = [
        _normalize_keyword_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", str(query or "").lower())
        if len(token) >= KEYWORD_MIN_LEN
    ]
    core_terms = [token for token in tokens if token not in STOPWORDS and token not in CORE_TERM_STOPWORDS]
    return list(dict.fromkeys(core_terms))


def _extract_document_selection_keywords(query: str) -> list[str]:
    tokens = _extract_query_keywords(query)
    meaningful = [token for token in tokens if token not in GENERIC_QUERY_WORDS]
    return list(dict.fromkeys(meaningful))


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _score_document_for_query(
    query_keywords: list[str],
    raw_query_tokens: set[str],
    document_name: str,
    doc_preview: str,
    metadata_preview: str,
    semantic_similarity: float,
) -> tuple[float, bool, bool]:
    if not query_keywords:
        return max(0.0, semantic_similarity), False, False

    normalized_name = _normalize_keyword_text(str(document_name or "").replace("_", " "))
    normalized_preview = _normalize_keyword_text(str(doc_preview or ""))
    normalized_meta = _normalize_keyword_text(str(metadata_preview or ""))
    combined_text = f"{normalized_name} {normalized_preview} {normalized_meta}".strip()
    combined_tokens = {
        _normalize_keyword_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", combined_text)
        if len(token) >= 2
    }

    filename_matches = sum(1 for keyword in query_keywords if keyword in normalized_name)
    preview_matches = sum(
        1
        for keyword in query_keywords
        if keyword in normalized_preview or keyword in normalized_meta
    )
    total_meaningful_matches = filename_matches + preview_matches

    generic_hits = sum(1 for token in combined_tokens if token in GENERIC_MATCH_WORDS)
    query_generic_hits = sum(1 for token in raw_query_tokens if token in GENERIC_MATCH_WORDS)
    only_generic_match = total_meaningful_matches == 0 and generic_hits > 0 and query_generic_hits > 0

    multiple_keywords_bonus = 2.0 if total_meaningful_matches >= 2 else 0.0

    domain_query = any(token in raw_query_tokens for token in DOMAIN_QUERY_TRIGGERS)
    domain_doc_hits = sum(1 for token in DOMAIN_BOOST_TERMS if token in combined_tokens)
    domain_boost = (2.0 * float(domain_doc_hits)) if domain_query and domain_doc_hits > 0 else 0.0

    generic_only_penalty = -2.0 if only_generic_match else 0.0

    keyword_score = (3.0 * float(filename_matches)) + (3.0 * float(preview_matches))
    semantic_score = max(0.0, semantic_similarity) * 1.5
    score = keyword_score + multiple_keywords_bonus + domain_boost + generic_only_penalty + semantic_score

    # Hard filter: meaningful keyword presence required and reject generic-only matches.
    passes_filter = total_meaningful_matches > 0 and not only_generic_match
    return score, passes_filter, only_generic_match


def _select_relevant_documents(
    collection: Any,
    query: str,
    embed_model: SentenceTransformer,
    top_n: int = 1,
) -> list[dict[str, str]]:
    records = collection.get(include=["metadatas", "documents"])
    metadatas = records.get("metadatas") or []
    documents = records.get("documents") or []

    query_keywords = _extract_document_selection_keywords(query)
    raw_query_tokens = {
        _normalize_keyword_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", str(query or "").lower())
        if token
    }
    print(f"[RAG] Query meaningful keywords: {query_keywords}")

    query_embedding = embed_model.encode(query, show_progress_bar=False).tolist()

    doc_catalog: dict[str, dict[str, str]] = {}
    for index, metadata in enumerate(metadatas):
        if not isinstance(metadata, dict):
            continue
        doc_id = str(metadata.get("doc_id", "")).strip()
        if not doc_id:
            continue
        file_name = str(metadata.get("file", "")).strip() or "Uploaded Doc"
        raw_doc = documents[index] if index < len(documents) else ""
        doc_text = " ".join(str(raw_doc or "").split()).strip()[:200]

        metadata_fields = []
        for key in ("title", "name", "author", "keywords", "subject", "description"):
            if key in metadata and metadata.get(key):
                metadata_fields.append(str(metadata.get(key)))
        metadata_preview = " ".join(metadata_fields).strip()[:200]

        if doc_id not in doc_catalog:
            doc_catalog[doc_id] = {
                "name": file_name,
                "doc_preview": doc_text,
                "metadata_preview": metadata_preview,
            }
            continue

        # Keep the first useful preview snippets for scoring.
        if not doc_catalog[doc_id]["doc_preview"] and doc_text:
            doc_catalog[doc_id]["doc_preview"] = doc_text
        if not doc_catalog[doc_id]["metadata_preview"] and metadata_preview:
            doc_catalog[doc_id]["metadata_preview"] = metadata_preview

    scored: list[tuple[float, bool, bool, str, str]] = []
    for doc_id, details in doc_catalog.items():
        doc_name = details.get("name", "Uploaded Doc")
        doc_preview = details.get("doc_preview", "")
        metadata_preview = details.get("metadata_preview", "")
        semantic_source = f"{doc_name} {doc_preview} {metadata_preview}".strip()
        semantic_embedding = embed_model.encode(semantic_source or doc_name, show_progress_bar=False).tolist()
        semantic_similarity = _cosine_similarity(query_embedding, semantic_embedding)
        score, passes_filter, only_generic_match = _score_document_for_query(
            query_keywords=query_keywords,
            raw_query_tokens=raw_query_tokens,
            document_name=doc_name,
            doc_preview=doc_preview,
            metadata_preview=metadata_preview,
            semantic_similarity=semantic_similarity,
        )
        scored.append((score, passes_filter, only_generic_match, doc_id, doc_name))

    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)

    filtered = [item for item in scored if item[1]]
    chosen = filtered[:top_n]
    if not chosen and scored:
        # Fallback rule: if hard filter rejects all docs, use best semantic/overall match.
        print("[RAG] Warning: No document passed meaningful-keyword filter, falling back to best semantic match")
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [scored[0]]
    chosen = chosen[:top_n]

    selected = [{"doc_id": doc_id, "name": doc_name} for _, _, _, doc_id, doc_name in chosen]
    selected_names = [item["name"] for item in selected]
    print(f"[RAG] Selected documents: {selected_names}")
    return selected


def _normalize_chunk_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _is_repetitive_chunk(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())
    if len(tokens) < 24:
        return False

    # Detect repeated phrase domination (e.g., repeated header/title lines).
    trigrams = [" ".join(tokens[i : i + 3]) for i in range(0, len(tokens) - 2)]
    if not trigrams:
        return False
    counts: dict[str, int] = {}
    for phrase in trigrams:
        counts[phrase] = counts.get(phrase, 0) + 1
    max_repeat = max(counts.values()) if counts else 0
    return max_repeat >= 3 and (float(max_repeat) / float(max(1, len(trigrams)))) >= 0.30


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
    return {
        _normalize_keyword_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2
    }


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks: list[str] = []
    cleaned = str(text or "")
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        end = start + chunk_size
        piece = cleaned[start:end]
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def _keyword_query_tokens(query: str) -> list[str]:
    keywords = _extract_query_keywords(query)
    if keywords:
        return keywords
    return [token for token in re.findall(r"[a-zA-Z0-9]+", str(query or "").lower()) if len(token) >= 3]


def _tokenize_for_bm25(text: str) -> list[str]:
    return [
        _normalize_keyword_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())
        if len(token) >= 2
    ]


def _length_quality_score(text: str) -> float:
    words = len(str(text or "").split())
    if words < 12:
        return -0.8
    if words < 20:
        return -0.3
    return min(1.0, float(words) / 120.0)


def _hybrid_rerank_score(
    query_terms: set[str],
    text: str,
    vector_score: float,
    bm25_score: float,
) -> float:
    tokens = _tokenize(text)
    keyword_overlap = len(query_terms & tokens)
    return (1.4 * float(vector_score)) + (1.0 * float(bm25_score)) + (0.35 * float(keyword_overlap)) + _length_quality_score(text)


def _has_generic_section_phrase(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(phrase in lowered for phrase in ["introduction", "conclusion", "summary"])


def _rerank_score(semantic_score: float, chunk_text: str, query_terms: set[str]) -> float:
    # Semantic similarity is primary; keyword overlap only provides a light boost.
    chunk_tokens = _tokenize(chunk_text)
    keyword_hits = len(query_terms & chunk_tokens)
    return float(semantic_score) + (0.1 * float(keyword_hits))


def _passes_explanatory_hard_rule(text: str) -> bool:
    lowered = f" {str(text or '').lower()} "
    tokens = _tokenize(text)
    has_verb = any(verb in tokens for verb in EXPLANATORY_VERBS)
    is_instruction = any(marker in lowered for marker in INSTRUCTION_MARKERS)
    symbol_ratio = len(re.findall(r"[^\w\s]", str(text or ""))) / float(max(1, len(str(text or ""))))
    numeric_like_only = (_digit_ratio(text) > 0.40 or symbol_ratio > 0.30) and not has_verb
    return has_verb and (not is_instruction) and (not numeric_like_only)


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
    has_explanatory_verb = any(verb in _tokenize(text) for verb in EXPLANATORY_VERBS)
    has_heavy_rulebook = rulebook_hits >= 2
    has_teaching_signal = teaching_hits >= 1 or conceptual_hits >= 2 or has_explanatory_verb
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
    tokens = [
        token
        for token in cleaned.split()
        if token and token not in STOPWORDS and token not in NON_CONTENT_QUERY_WORDS and len(token) >= 3
    ]
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
    # Reuse the single global embedder managed by db.py.
    return get_embedder()


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> RetrievalResult:
    collection = get_collection()
    total_chunks = int(collection.count())
    print(f"[RAG] Total chunks: {total_chunks}")

    if total_chunks <= 0:
        return {"chunks": [], "context": "No relevant context found in documents."}

    query_terms = set(_extract_query_keywords(query))
    query_keywords = list(query_terms)
    print(f"[RAG] Keywords: {query_keywords}")
    query_tokens = _keyword_query_tokens(query)
    embed_model = _get_embed_model()
    query_embedding = embed_model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    vector_k = min(max(5, int(top_k)), total_chunks)
    where = {"doc_id": document_id} if document_id else None
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=vector_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = (vector_results.get("documents") or [[]])[0] if vector_results.get("documents") else []
    metadatas = (vector_results.get("metadatas") or [[]])[0] if vector_results.get("metadatas") else []
    distances = (vector_results.get("distances") or [[]])[0] if vector_results.get("distances") else []

    retrieved_count = len(docs)
    print(f"[RAG] Retrieved (vector): {retrieved_count}")

    candidates_by_sig: dict[str, dict[str, Any]] = {}

    def _register_candidate(
        text: str,
        metadata: dict[str, Any],
        vector_score: float = 0.0,
        bm25_score: float = 0.0,
        chunk_idx: int | None = None,
    ) -> None:
        clean_text = _normalize_chunk_text(_clean_broken_sentences(text))
        if not clean_text:
            return

        for phrase in TITLE_NOISE_PHRASES:
            clean_text = re.sub(re.escape(phrase), " ", clean_text, flags=re.IGNORECASE)
        clean_text = _normalize_chunk_text(clean_text)
        if len(clean_text.split()) < 12:
            return

        signature = re.sub(r"\s+", " ", clean_text.lower())
        if not signature:
            return

        page_val = metadata.get("page") if isinstance(metadata, dict) else None
        page = page_val if isinstance(page_val, int) else int(page_val) if isinstance(page_val, str) and page_val.isdigit() else None
        file_name = str((metadata or {}).get("file", "unknown.pdf"))
        doc_id = str((metadata or {}).get("doc_id", ""))

        existing = candidates_by_sig.get(signature)
        if existing:
            existing["vector_score"] = max(float(existing.get("vector_score", 0.0)), float(vector_score))
            existing["bm25_score"] = max(float(existing.get("bm25_score", 0.0)), float(bm25_score))
            return

        enriched_meta = dict(metadata or {})
        enriched_meta["source"] = file_name
        enriched_meta["chunk_id"] = (
            str(chunk_idx)
            if chunk_idx is not None
            else f"{doc_id or 'doc'}:{abs(hash(signature)) % 1000000}"
        )

        candidates_by_sig[signature] = {
            "text": clean_text,
            "file": file_name,
            "doc_id": doc_id,
            "page": page,
            "metadata": enriched_meta,
            "vector_score": float(vector_score),
            "bm25_score": float(bm25_score),
        }

    for idx, raw_chunk in enumerate(docs):
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        distance_val = distances[idx] if idx < len(distances) else None
        semantic_score = 0.0
        if isinstance(distance_val, (int, float)):
            semantic_score = max(0.0, 1.0 - (float(distance_val) / 2.0))

        vector_windows = chunk_text(str(raw_chunk or ""), chunk_size=500, overlap=100)
        for window_idx, window in enumerate(vector_windows):
            _register_candidate(
                text=window,
                metadata=metadata,
                vector_score=semantic_score,
                bm25_score=0.0,
                chunk_idx=window_idx,
            )

    all_results = collection.get(where=where, include=["documents", "metadatas"])
    all_docs = all_results.get("documents") or []
    all_metas = all_results.get("metadatas") or []

    bm25_corpus: list[str] = []
    bm25_meta: list[dict[str, Any]] = []
    for idx, raw_doc in enumerate(all_docs):
        metadata = all_metas[idx] if idx < len(all_metas) and isinstance(all_metas[idx], dict) else {}
        for window_idx, window in enumerate(chunk_text(str(raw_doc or ""), chunk_size=500, overlap=100)):
            cleaned_window = _normalize_chunk_text(window)
            if len(cleaned_window.split()) < 12:
                continue
            bm25_corpus.append(cleaned_window)
            bm25_meta.append({"meta": metadata, "chunk_idx": window_idx})

    if bm25_corpus and query_tokens:
        tokenized_chunks = [chunk.split() for chunk in bm25_corpus]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(query_tokens)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
        max_bm25 = max([bm25_scores[i] for i in top_bm25_indices], default=0.0)

        for bm25_idx in top_bm25_indices:
            raw_score = float(bm25_scores[bm25_idx])
            normalized_bm25 = (raw_score / max_bm25) if max_bm25 > 0 else 0.0
            _register_candidate(
                text=bm25_corpus[bm25_idx],
                metadata=bm25_meta[bm25_idx]["meta"],
                vector_score=0.0,
                bm25_score=normalized_bm25,
                chunk_idx=int(bm25_meta[bm25_idx]["chunk_idx"]),
            )

    combined_candidates = list(candidates_by_sig.values())
    if not combined_candidates:
        return {"chunks": [], "context": "No relevant context found in documents."}

    reranked: list[tuple[int, float, int, RetrievedChunk]] = []
    for candidate in combined_candidates:
        text = str(candidate.get("text", "")).strip()
        if len(text.split()) < 12:
            continue
        if _is_repetitive_chunk(text):
            continue
        if _has_generic_section_phrase(text):
            continue

        score = _hybrid_rerank_score(
            query_terms=query_terms,
            text=text,
            vector_score=float(candidate.get("vector_score", 0.0)),
            bm25_score=float(candidate.get("bm25_score", 0.0)),
        )

        lowered_text = text.lower()
        keyword_hits = sum(1 for word in query_keywords if word in lowered_text)
        print(f"[RAG] Chunk keyword hits: {keyword_hits} | {text[:90]}")

        # Strong keyword boost for concept-specific queries.
        score += float(keyword_hits) * 2.0

        chunk_len = len(text.split())

        reranked.append(
            (
                keyword_hits,
                score,
                chunk_len,
                {
                    "text": text,
                    "file": str(candidate.get("file", "unknown.pdf")),
                    "doc_id": str(candidate.get("doc_id", "")),
                    "page": candidate.get("page") if isinstance(candidate.get("page"), int) else None,
                    "metadata": dict(candidate.get("metadata", {})),
                },
            )
        )

    # Priority sorting: keyword hits > hybrid score > chunk length.
    reranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    # Combine vector+BM25 candidates and keep a compact clean set.
    selected_chunks: list[RetrievedChunk] = []
    keyword_first = [item for item in reranked if item[0] > 0]
    keyword_last = [item for item in reranked if item[0] == 0]
    ordered_candidates = keyword_first + keyword_last

    for _, _, _, chunk in ordered_candidates[:7]:
        text = _normalize_chunk_text(chunk.get("text", ""))
        if len(text.split()) < 12:
            continue
        if any(_is_near_duplicate(text, existing.get("text", "")) for existing in selected_chunks):
            continue
        selected_chunks.append(chunk)
        if len(selected_chunks) >= 5:
            break

    if not selected_chunks and reranked:
        selected_chunks = [reranked[0][3]]

    print("[RAG] FINAL CHUNKS AFTER HYBRID RERANK:")
    for idx, chunk in enumerate(selected_chunks, start=1):
        print(f"[RAG] Chunk {idx}: {chunk.get('text', '')[:120]}")

    context = "\n\n".join(chunk.get("text", "").strip() for chunk in selected_chunks if chunk.get("text", "").strip())
    return {"chunks": selected_chunks, "context": context}
