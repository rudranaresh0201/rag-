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
You are an academic assistant.

Answer using ONLY the provided context.

Structure:
1. Short summary (2-3 lines)
2. Key points (bullet points)

Guidelines:
- Be clear and factual
- Use simple academic language
- Do NOT repeat the context

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

        if not context.strip():
            print("FINAL ANSWER READY")
            return "No relevant information found in the provided document."

        prompt = _build_prompt(query=query, context=context)
        tokenizer, model = _get_model_and_tokenizer()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        logger.info("LLM prompt_length=%s", len(prompt))
        logger.info("LLM token_count=%s", inputs["input_ids"].shape[-1])

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=250,
            do_sample=False,
            temperature=0.2,
            max_time=GENERATION_MAX_TIME,
        )

        prompt_token_count = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_token_count:]
        output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("=== RAW MODEL OUTPUT ===")
        print(output)

        answer = output.replace("\n\n", "\n").strip()

        print("CLEAN ANSWER:", answer)
        print("FINAL ANSWER:", answer)

        logger.info("LLM output_length=%s", len(answer))

        if not answer or len(answer.strip()) < 10:
            print("WARNING: Empty or weak output")
            print("FINAL ANSWER READY")
            return "No relevant information found in the provided document."

        print("FINAL ANSWER READY")
        return answer
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        print("GENERATION ERROR:", exc)
        print("FINAL ANSWER READY")
        return "Error generating response"
