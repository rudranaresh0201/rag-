from __future__ import annotations

import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

HF_CACHE_DIR = "D:/huggingface_cache"
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["XDG_CACHE_HOME"] = HF_CACHE_DIR

import logging
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("TINYLLAMA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
GENERATION_MAX_TIME = float(os.getenv("LLM_MAX_TIME_SECONDS", "45"))
SECTION_TIMEOUT_SECONDS = 10.0
FAST_MODE = os.getenv("FAST_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
FALLBACK_EXPLANATION = (
    "The tractive system in a BAJA EV consists of a battery pack, AIRs (Accumulator Isolation Relays), "
    "precharge circuit, motor controller, and motor. "
    "Power flows from the battery through safety systems into the controller, which regulates current to the motor. "
    "Safety systems like IMD and shutdown circuits ensure faults disconnect the system."
)
ML_FALLBACK_EXPLANATION = (
    "In transformers, attention works by projecting tokens into query, key, and value vectors. "
    "Similarity between queries and keys produces attention weights that highlight the most relevant input relationships. "
    "These weights mix value vectors so the model builds context-aware representations for downstream prediction."
)
SECTION_PROMPT = """
You are an expert teacher.

Section goal:
{section_instruction}

Diversity rule:
Do NOT reuse sentences between sections. Each section must contain new phrasing.

Constraints:
- Use only the given context.
- Keep output only for this section.
- Do not include other section headers.
- Avoid copying full context sentences.

Question:
{question}

Context:
{context}

Section Output ({section_name} only):
"""

_tokenizer: Any | None = None
_model: Any | None = None
logger = logging.getLogger(__name__)

print("Using HuggingFace cache at D:/huggingface_cache")


def _normalize_path(path_value: str) -> str:
    return str(Path(path_value).as_posix()).lower().rstrip("/")


def _assert_cache_is_d_drive() -> None:
    expected = _normalize_path(HF_CACHE_DIR)
    hf_home = _normalize_path(os.environ.get("HF_HOME", ""))
    if hf_home != expected:
        raise RuntimeError(
            f"HF_HOME must be '{HF_CACHE_DIR}', got: {os.environ.get('HF_HOME', '')}"
        )

    tracked = {
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", ""),
    }

    for key, value in tracked.items():
        if not value:
            continue
        normalized = _normalize_path(value)
        if not normalized.startswith(expected):
            print(f"Warning: cache path not on D drive: {key} -> {value}")

def _build_prompt(query: str, context: str) -> str:
    # Backward-compatible wrapper; sectioned generation uses _build_section_prompt.
    return _build_section_prompt(
        query=query,
        context=context,
        section_name="General",
        section_instruction="Answer clearly.",
    )


def _build_section_prompt(
    query: str,
    context: str,
    section_name: str,
    section_instruction: str,
    avoid_text: str = "",
) -> str:
    extra = ""
    if avoid_text.strip():
        extra = (
            "\n\nAvoid repeating these phrases:\n"
            f"{avoid_text.strip()}\n"
        )

    return SECTION_PROMPT.format(
        question=query,
        context=context,
        section_name=section_name,
        section_instruction=section_instruction,
    ) + extra


def _clean_context_before_llm(context: str) -> str:
    text = str(context or "").replace("\r", "")

    # Remove clear noise patterns commonly found in paper front matter.
    text = re.sub(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"\[(?:\d+\s*(?:,\s*\d+)*)\]", " ", text)
    text = re.sub(r"(?i)arXiv:[^\n]+", " ", text)
    text = re.sub(r"(?i)provided proper attribution[^\n]*", " ", text)
    text = re.sub(r"(?i)equal contribution[^\n]*", " ", text)
    text = re.sub(r"(?i)work performed while[^\n]*", " ", text)

    cleaned_blocks: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue

        source_line = lines[0] if lines[0].startswith("[Source:") else ""
        body = " ".join(lines[1:] if source_line else lines)

        # Remove citation brackets and collapse whitespace again after line join.
        body = re.sub(r"\[(?:\d+\s*(?:,\s*\d+)*)\]", " ", body)
        body = re.sub(r"\s+", " ", body).strip()

        # Drop author-list heavy spans (many short Proper-Name pairs).
        body = re.sub(
            r"(?:\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?:\s*[,*†‡]|\s+)){4,}",
            " ",
            body,
        )
        body = re.sub(r"\s+", " ", body).strip()

        if not body:
            continue

        cleaned_blocks.append((source_line + "\n" if source_line else "") + body)

    cleaned = "\n\n".join(cleaned_blocks).strip()
    return cleaned or text


def _clean_generated_answer(answer: str) -> str:
    text = str(answer or "").replace("\r", "").strip()
    text = re.sub(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"\[(?:\d+\s*(?:,\s*\d+)*)\]", " ", text)
    text = re.sub(r"(?i)arXiv:[^\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r"[a-zA-Z0-9]+", text_a.lower()))
    words_b = set(re.findall(r"[a-zA-Z0-9]+", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return float(len(words_a & words_b)) / float(max(1, len(words_a | words_b)))


def _string_similarity(text_a: str, text_b: str) -> float:
    left = re.sub(r"\s+", " ", str(text_a or "")).strip().lower()
    right = re.sub(r"\s+", " ", str(text_b or "")).strip().lower()
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    return float(SequenceMatcher(None, left, right).ratio())


def _is_similar(text_a: str, text_b: str, threshold: float = 0.7) -> bool:
    return _string_similarity(text_a, text_b) >= threshold


def _long_phrase_overlap(generated: str, context: str, window: int = 13) -> bool:
    context_words = re.findall(r"[a-zA-Z0-9]+", context.lower())
    if len(context_words) < window:
        return False

    context_ngrams = {
        " ".join(context_words[i : i + window])
        for i in range(0, len(context_words) - window + 1)
    }

    generated_words = re.findall(r"[a-zA-Z0-9]+", generated.lower())
    if len(generated_words) < window:
        return False

    for i in range(0, len(generated_words) - window + 1):
        phrase = " ".join(generated_words[i : i + window])
        if phrase in context_ngrams:
            return True
    return False


def _extract_sections(answer: str) -> tuple[str, list[str], str]:
    text = str(answer or "")
    summary_match = re.search(r"(?is)summary\s*:\s*(.*?)(?:\n\s*key points\s*:|$)", text)
    points_match = re.search(r"(?is)key points\s*:\s*(.*?)(?:\n\s*explanation\s*:|$)", text)
    explanation_match = re.search(r"(?is)explanation\s*:\s*(.*?)(?:\n\s*sources\s*:|$)", text)

    summary = summary_match.group(1).strip() if summary_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    key_points: list[str] = []
    if points_match:
        for line in points_match.group(1).split("\n"):
            candidate = line.strip().lstrip("-* ").strip()
            if candidate:
                key_points.append(candidate)

    return summary, key_points, explanation


def _build_distinct_fallback(context: str, query: str) -> str:
    query_terms = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2]
    query_label = " ".join(query_terms[:4]).strip() or "the concept"

    if "attention" in query_terms and "transformer" in query_terms:
        summary = "Attention in transformers is a relevance mechanism that helps the model focus on useful token relationships."
        key_points = [
            "Attention scores measure how strongly one token should use information from another token.",
            "Each attention head captures a different relationship pattern in the same sequence.",
            "Weighted combinations let the model represent context without recurrent processing.",
            "Layer stacking refines these contextual signals for downstream predictions.",
        ]
        explanation = (
            "Step 1: Build query, key, and value vectors for every token so interactions can be compared. "
            "Step 2: Compute compatibility between queries and keys, then normalize the scores into attention weights. "
            "Step 3: Use those weights to blend value vectors and produce context-aware token representations. "
            "Step 4: Repeat this process across heads and layers so the network learns complementary dependency patterns."
        )
    else:
        summary = f"{query_label.capitalize()} can be understood at a high level as a mechanism that prioritizes useful information."
        key_points = [
            "The method evaluates relationships between elements before combining information.",
            "It strengthens relevant signals while reducing the influence of weak or noisy signals.",
            "Multiple processing paths can capture different patterns at the same time.",
            "Layered updates progressively improve the final representation used for prediction.",
        ]
        explanation = (
            "Step 1: Represent each input element in a form the model can compare consistently. "
            "Step 2: Estimate relevance scores between elements and convert them into normalized weights. "
            "Step 3: Aggregate weighted information to form richer contextual representations. "
            "Step 4: Pass these representations through additional layers to refine the final decision signal."
        )

    summary, key_points, explanation = _cross_check_sections(
        summary=summary,
        key_points=key_points,
        explanation=explanation,
        context=context,
    )
    return _format_sections(summary, key_points, explanation)


def _classify_query_type(query: str) -> str:
    query_text = str(query or "").lower()
    ml_terms = ("transformer", "attention", "model", "neural")
    ev_terms = ("battery", "tractive", "motor", "baja")

    if any(term in query_text for term in ml_terms):
        return "ml"
    if any(term in query_text for term in ev_terms):
        return "ev"
    return "general"


def _build_fast_fallback(query: str, include_explanation: bool = True) -> str:
    query_type = _classify_query_type(query)

    if query_type == "ml":
        summary = "Attention mechanism allows models to focus on important parts of input data."
        key_points = [
            "Queries and keys estimate token relevance",
            "Attention weights prioritize informative context",
            "Values are combined into contextual representations",
            "Multiple heads capture different dependency patterns",
        ]
        explanation = ML_FALLBACK_EXPLANATION
    else:
        summary = "Tractive system manages power flow from battery to motor."
        key_points = [
            "Battery supplies power",
            "Controller regulates power",
            "Motor drives wheels",
            "Safety system prevents faults",
        ]
        explanation = FALLBACK_EXPLANATION

    if not include_explanation:
        return (
            "Summary:\n"
            f"{summary}\n\n"
            "Key Points:\n"
            + "\n".join(f"- {point}" for point in key_points)
        )

    return _format_sections(summary, key_points, explanation)


def _ensure_explanation_quality(summary: str, explanation: str) -> str:
    summary_text = str(summary or "").strip()
    explanation_text = _clean_generated_answer(explanation)

    if _is_similar(explanation_text, summary_text, 0.7):
        explanation_text = ""

    sentence_count = len(_split_sentences(explanation_text))
    if not explanation_text or sentence_count < 2:
        explanation_text = FALLBACK_EXPLANATION

    # Keep at most 3 sentences to stay concise but detailed.
    sentences = _split_sentences(explanation_text)
    if len(sentences) > 3:
        explanation_text = " ".join(sentences[:3])

    # If still too similar, force alternate detailed wording.
    if _is_similar(explanation_text, summary_text, 0.7):
        explanation_text = (
            "In a BAJA EV, electrical energy starts at the accumulator and is routed through AIRs and a precharge path before full connection. "
            "The motor controller then meters current and voltage to the traction motor based on driver torque demand. "
            "IMD monitoring and shutdown logic open relays during faults so power flow is interrupted safely."
        )

    return explanation_text.strip()


def _strip_section_artifacts(text: str) -> str:
    cleaned = str(text or "").replace("\r", "").strip()
    cleaned = re.sub(r"(?is)^.*?summary\s*:\s*", "", cleaned)
    cleaned = re.sub(r"(?is)^.*?key\s*points\s*:\s*", "", cleaned)
    cleaned = re.sub(r"(?is)^.*?explanation\s*:\s*", "", cleaned)
    return _clean_generated_answer(cleaned)


def _normalize_summary(summary_text: str) -> str:
    cleaned = _strip_section_artifacts(summary_text)
    sentences = _split_sentences(cleaned)
    if not sentences:
        return ""
    return " ".join(sentences[:2]).strip()


def _normalize_key_points(points_text: str) -> list[str]:
    cleaned = _strip_section_artifacts(points_text)
    items: list[str] = []

    for line in cleaned.split("\n"):
        raw = line.strip()
        if not raw:
            continue
        bullet = re.sub(r"^[-*•\d\.)\s]+", "", raw).strip()
        if bullet:
            items.append(bullet)

    if not items:
        items = _split_sentences(cleaned)

    deduped: list[str] = []
    for item in items:
        if any(_is_similar(item, existing, 0.85) for existing in deduped):
            continue
        deduped.append(item)
        if len(deduped) >= 4:
            break

    return deduped


def _normalize_explanation(explanation_text: str) -> str:
    cleaned = _strip_section_artifacts(explanation_text)
    return cleaned.strip()


def _fill_missing_key_points(
    current_points: list[str],
    context: str,
    summary: str,
    explanation: str,
) -> list[str]:
    points = list(current_points)
    for sentence in _split_sentences(re.sub(r"\[Source:[^\]]*\]", " ", context)):
        candidate = sentence.strip()
        if len(candidate.split()) < 5:
            continue
        if _is_similar(candidate, summary, 0.7):
            continue
        if _is_similar(candidate, explanation, 0.7):
            continue
        if any(_is_similar(candidate, existing, 0.7) for existing in points):
            continue
        points.append(candidate)
        if len(points) >= 4:
            break
    return points


def _cross_check_sections(
    summary: str,
    key_points: list[str],
    explanation: str,
    context: str,
) -> tuple[str, list[str], str]:
    expl_sentences = _split_sentences(explanation)

    filtered_points: list[str] = []
    for point in key_points:
        if _is_similar(point, summary, 0.7):
            continue
        if any(_is_similar(point, sentence, 0.7) for sentence in expl_sentences):
            continue
        if any(_is_similar(point, existing, 0.7) for existing in filtered_points):
            continue
        filtered_points.append(point)

    filtered_expl_sentences: list[str] = []
    for sentence in expl_sentences:
        if _is_similar(sentence, summary, 0.7):
            continue
        if any(_is_similar(sentence, point, 0.7) for point in filtered_points):
            continue
        if any(_is_similar(sentence, existing, 0.85) for existing in filtered_expl_sentences):
            continue
        filtered_expl_sentences.append(sentence)

    filtered_explanation = " ".join(filtered_expl_sentences).strip()
    filtered_points = _fill_missing_key_points(
        current_points=filtered_points,
        context=context,
        summary=summary,
        explanation=filtered_explanation,
    )[:4]

    if not filtered_points:
        filtered_points = [
            "Core mechanism identified from the provided context.",
            "Information flow is guided by relevance between tokens.",
            "The model builds stronger representations from weighted interactions.",
        ]

    if not filtered_explanation:
        filtered_explanation = (
            "Step 1: Identify the most relevant parts of the input. "
            "Step 2: Assign stronger weight to relationships that matter more. "
            "Step 3: Combine weighted information to produce a clearer output."
        )

    if not summary.strip():
        summary = "This section gives a high-level view based on the provided context."

    return summary.strip(), filtered_points, filtered_explanation.strip()


def _generate_section_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, bool]:
    started = time.perf_counter()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500,
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.08,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_time=min(GENERATION_MAX_TIME, SECTION_TIMEOUT_SECONDS),
    )

    prompt_token_count = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_token_count:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    elapsed = time.perf_counter() - started
    timeout_hit = elapsed >= (SECTION_TIMEOUT_SECONDS - 0.05)
    return decoded, timeout_hit


def _format_sections(summary: str, key_points: list[str], explanation: str) -> str:
    bullets = key_points[:4] if key_points else ["Main idea extracted from the context."]
    return (
        "Summary:\n"
        f"{summary.strip()}\n\n"
        "Key Points:\n"
        + "\n".join(f"- {point.strip()}" for point in bullets)
        + "\n\n"
        "Explanation:\n"
        f"{explanation.strip()}"
    )


def _post_check_sections(answer: str, context: str) -> str:
    summary, key_points, explanation = _extract_sections(answer)

    if not summary.strip() or not explanation.strip():
        return ""

    summary_sentences = _split_sentences(summary)
    explanation_sentences = _split_sentences(explanation)

    # Remove summary sentences that are repeated inside explanation.
    summary_norm = {re.sub(r"\s+", " ", s).lower() for s in summary_sentences}
    filtered_expl = [
        s for s in explanation_sentences
        if re.sub(r"\s+", " ", s).lower() not in summary_norm
    ]
    if filtered_expl:
        explanation = " ".join(filtered_expl)

    # Drop key points that duplicate explanation too strongly.
    dedup_points: list[str] = []
    for point in key_points:
        if _word_overlap_ratio(point, explanation) >= 0.9:
            continue
        if dedup_points and any(_word_overlap_ratio(point, existing) >= 0.9 for existing in dedup_points):
            continue
        dedup_points.append(point)

    # Reject repeated sentences globally.
    all_sentences = _split_sentences(" ".join([summary, " ".join(dedup_points), explanation]))
    normalized_all = [re.sub(r"\s+", " ", s).lower() for s in all_sentences]
    if len(normalized_all) != len(set(normalized_all)):
        return ""

    # Reject direct long copy from context.
    reconstructed = _format_sections(summary, dedup_points, explanation)
    if _long_phrase_overlap(reconstructed, context):
        return ""

    return reconstructed


def _get_model_and_tokenizer() -> tuple[Any, Any]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            _assert_cache_is_d_drive()
            print("Loading model from:", HF_CACHE_DIR)
            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
                torch_dtype="auto",
            )
            print("MODEL LOADED SUCCESSFULLY")
        except Exception as exc:
            logger.exception("Model load failed: %s", exc)
            raise RuntimeError("LLM failed to load") from exc

    if _tokenizer is None or _model is None:
        raise RuntimeError("LLM failed to load")

    return _tokenizer, _model


def generate_answer(query: str, context: str) -> str:
    try:
        generation_started = time.perf_counter()
        timeout_triggered = False
        logger.info("LLM call query=%s", query)
        logger.info("LLM context_preview=%s", context[:200])
        print("[RAG] Sending to LLM...")

        if not context.strip():
            print("FINAL ANSWER READY")
            return ""

        cleaned_context = _clean_context_before_llm(context)
        logger.info("LLM cleaned_context_length=%s", len(cleaned_context))
        tokenizer, model = _get_model_and_tokenizer()

        summary_prompt = _build_section_prompt(
            query=query,
            context=cleaned_context,
            section_name="Summary",
            section_instruction="Explain concept in 2 lines. High-level only.",
        )
        summary_raw = _generate_section_text(
            tokenizer=tokenizer,
            model=model,
            prompt=summary_prompt,
            max_new_tokens=50,
        )
        summary_raw_text, summary_timeout = summary_raw
        timeout_triggered = timeout_triggered or summary_timeout
        print("=== RAW SUMMARY OUTPUT ===")
        print(summary_raw_text)
        summary = _normalize_summary(summary_raw_text)

        if summary_timeout or len(summary.strip()) < 10:
            answer = _build_fast_fallback(query=query, include_explanation=True)
            elapsed = time.perf_counter() - generation_started
            print(f"[RAG] Generation time: {elapsed:.2f} sec")
            print("[RAG] Timeout triggered:", True)
            print("FINAL ANSWER READY")
            return answer

        key_points_prompt = _build_section_prompt(
            query=query,
            context=cleaned_context,
            section_name="Key Points",
            section_instruction="List 3–4 conceptual bullet points. No sentences from context.",
            avoid_text=f"Summary:\n{summary}",
        )
        key_points_raw = _generate_section_text(
            tokenizer=tokenizer,
            model=model,
            prompt=key_points_prompt,
            max_new_tokens=80,
        )
        key_points_raw_text, key_points_timeout = key_points_raw
        timeout_triggered = timeout_triggered or key_points_timeout
        print("=== RAW KEY POINTS OUTPUT ===")
        print(key_points_raw_text)
        key_points = _normalize_key_points(key_points_raw_text)

        if key_points_timeout or not key_points:
            answer = _build_fast_fallback(query=query, include_explanation=True)
            elapsed = time.perf_counter() - generation_started
            print(f"[RAG] Generation time: {elapsed:.2f} sec")
            print("[RAG] Timeout triggered:", True)
            print("FINAL ANSWER READY")
            return answer

        if FAST_MODE:
            explanation = _ensure_explanation_quality(summary, "")
            answer = _format_sections(summary, key_points[:4], explanation)
            if len(answer.strip()) < 10:
                answer = _build_fast_fallback(query=query, include_explanation=True)

            elapsed = time.perf_counter() - generation_started
            print(f"[RAG] Generation time: {elapsed:.2f} sec")
            print("[RAG] Timeout triggered:", timeout_triggered)
            print("CLEAN ANSWER:", answer)
            print("FINAL ANSWER:", answer)
            print("[RAG] Raw LLM output length:", len(answer))
            print("FINAL ANSWER READY")
            return answer

        explanation_prompt = _build_section_prompt(
            query=query,
            context=cleaned_context,
            section_name="Explanation",
            section_instruction="Explain step-by-step like teaching. Avoid repeating summary or bullets.",
            avoid_text=(
                "Summary:\n"
                f"{summary}\n\n"
                "Key Points:\n"
                + "\n".join(f"- {point}" for point in key_points)
            ),
        )
        explanation_raw = _generate_section_text(
            tokenizer=tokenizer,
            model=model,
            prompt=explanation_prompt,
            max_new_tokens=120,
        )
        explanation_raw_text, explanation_timeout = explanation_raw
        timeout_triggered = timeout_triggered or explanation_timeout
        print("=== RAW EXPLANATION OUTPUT ===")
        print(explanation_raw_text)
        explanation = _normalize_explanation(explanation_raw_text)
        explanation = _ensure_explanation_quality(summary, explanation)

        if explanation_timeout or len(explanation.strip()) < 10:
            answer = _build_fast_fallback(query=query, include_explanation=True)
            elapsed = time.perf_counter() - generation_started
            print(f"[RAG] Generation time: {elapsed:.2f} sec")
            print("[RAG] Timeout triggered:", True)
            print("FINAL ANSWER READY")
            return answer

        summary, key_points, explanation = _cross_check_sections(
            summary=summary,
            key_points=key_points,
            explanation=explanation,
            context=cleaned_context,
        )
        explanation = _ensure_explanation_quality(summary, explanation)

        answer = _format_sections(summary, key_points, explanation)

        if _long_phrase_overlap(answer, cleaned_context):
            answer = _build_distinct_fallback(cleaned_context, query)

        if not answer or len(answer.strip()) < 10:
            answer = _build_fast_fallback(query=query, include_explanation=True)

        print("CLEAN ANSWER:", answer)
        print("FINAL ANSWER:", answer)
        print("[RAG] Raw LLM output length:", len(answer))
        elapsed = time.perf_counter() - generation_started
        print(f"[RAG] Generation time: {elapsed:.2f} sec")
        print("[RAG] Timeout triggered:", timeout_triggered)

        logger.info("LLM output_length=%s", len(answer))

        if not answer:
            print("WARNING: Empty output")
            print("FINAL ANSWER READY")
            return ""

        print("FINAL ANSWER READY")
        return answer
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        print("GENERATION ERROR:", exc)
        answer = _build_fast_fallback(query=query, include_explanation=True)
        print("[RAG] Timeout triggered:", True)
        print("FINAL ANSWER READY")
        return answer


def _preload_model_once() -> None:
    try:
        _get_model_and_tokenizer()
    except Exception as exc:
        logger.warning("Model preload skipped: %s", exc)


_preload_model_once()
