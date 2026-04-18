from __future__ import annotations

import os
import re
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
PROMPT = """
You are an AI assistant.

You are given context extracted from documents.

Your job is to EXTRACT and EXPLAIN the answer from this context.

Rules:
- DO NOT decide whether context is relevant
- ALWAYS try to extract useful information from context
- Use whatever information is available
- Only say 'Insufficient information' if context is completely empty

Return your answer in exactly this format:
Summary:
<concise answer>

Key Points:
- <point 1>
- <point 2>

Explanation:
<detailed explanation grounded in the provided context>

Context:
{context}

Question:
{question}

Answer:
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
    return PROMPT.format(context=context, question=query)


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
        logger.info("LLM call query=%s", query)
        logger.info("LLM context_preview=%s", context[:200])
        print("[RAG] Sending to LLM...")

        if not context.strip():
            print("FINAL ANSWER READY")
            return ""

        prompt = _build_prompt(query=query, context=context)
        tokenizer, model = _get_model_and_tokenizer()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1500,
        )
        logger.info("LLM prompt_length=%s", len(prompt))
        logger.info("LLM token_count=%s", inputs["input_ids"].shape[-1])

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=380,
            do_sample=False,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_time=GENERATION_MAX_TIME,
        )

        prompt_token_count = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_token_count:]
        output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("=== RAW MODEL OUTPUT ===")
        print(output)

        answer = output.replace("\r", "").strip()
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        answer = answer.replace("\u0000", "").strip()

        # Drop obvious prompt-echo artifacts if present.
        answer = re.sub(r"(?is)^.*?Summary:\s*", "Summary:\n", answer) if "Summary:" in answer else answer

        print("CLEAN ANSWER:", answer)
        print("FINAL ANSWER:", answer)
        print("[RAG] Raw LLM output length:", len(answer))

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
        print("FINAL ANSWER READY")
        return ""
