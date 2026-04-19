"""
llm_router.py — Modular LLM provider router.

Provides a single public interface:

    generate_answer(query, context, provider="openai") -> str

Supported providers:
  - "openai"  : OpenAI ChatCompletion (default). Requires OPENAI_API_KEY env var.
  - "hf"      : Local HuggingFace model (TinyLlama fallback). Uses llm.py's pipeline.

The router is intentionally thin — provider implementations live in their own
private functions and share nothing except the common prompt template.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared prompt
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = """\
You are a highly reliable AI assistant designed for Retrieval-Augmented Generation (RAG).

Your job is to answer the user question using ONLY the provided context.

========================
CONTEXT:
{context}
========================

QUESTION:
{query}

========================
STRICT RULES:

1. Use ONLY relevant information from the context.
2. If context is noisy or irrelevant, IGNORE irrelevant parts.
3. If no useful information is found, respond EXACTLY:
    "Insufficient relevant information found in the provided documents."
4. NEVER output single letters, symbols, or incomplete answers.
5. ALWAYS generate a complete, meaningful explanation.
6. Keep answer concise but informative (5-8 lines max).

========================
OUTPUT FORMAT (MANDATORY):

Answer:
<clear explanation in simple terms>

Key Points:
- <point 1>
- <point 2>
- <point 3>

Confidence:
- High / Medium / Low

========================

Now generate the BEST possible answer.
"""


def _build_prompt(query: str, context: str) -> str:
    return _PROMPT_TEMPLATE.format(context=context[:2000], query=query)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------
def _generate_openai(query: str, context: str) -> str:
    """
    Call OpenAI ChatCompletion (gpt-3.5-turbo by default).
    Set OPENAI_API_KEY and optionally OPENAI_MODEL in environment.
    """
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    prompt = _build_prompt(query, context)

    print(f"[RAG] LLM provider=openai model={model}")
    logger.info("LLM provider=openai model=%s query=%s", model, query)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document analysis assistant. "
                    "Always extract and explain answers from the provided context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.1,
    )

    output = response.choices[0].message.content or ""
    output = output.strip()
    print(f"[RAG] OpenAI response length: {len(output)}")
    logger.info("LLM output_length=%s", len(output))
    return output


# ---------------------------------------------------------------------------
# HuggingFace / local model provider
# ---------------------------------------------------------------------------
def _generate_hf(query: str, context: str) -> str:
    """
    Delegate to the existing llm.generate_answer() which loads TinyLlama
    (or whatever TINYLLAMA_MODEL env var points to).
    """
    print("[RAG] LLM provider=hf (local model)")
    logger.info("LLM provider=hf query=%s", query)

    try:
        from .llm import generate_answer as hf_generate  # type: ignore
    except ImportError:
        from llm import generate_answer as hf_generate  # type: ignore

    return hf_generate(query=query, context=context)


# ---------------------------------------------------------------------------
# Public router interface
# ---------------------------------------------------------------------------
def generate_answer(
    query: str,
    context: str,
    provider: str | None = None,
) -> str:
    """
    Route to the appropriate LLM provider and return the raw text answer.

    provider resolution order:
      1. `provider` argument (explicit)
      2. LLM_PROVIDER environment variable
      3. Falls back to "openai" if OPENAI_API_KEY is set, else "hf"
    """
    if not context.strip():
        print("[RAG] Empty context — skipping LLM call")
        return ""

    # Resolve provider
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY", "").strip() else "hf"

    print(f"[RAG] Routing to LLM provider: {provider}")

    try:
        if provider == "openai":
            return _generate_openai(query, context)
        elif provider in ("hf", "huggingface", "local"):
            return _generate_hf(query, context)
        else:
            logger.warning("Unknown LLM provider '%s'; falling back to hf", provider)
            print(f"[RAG] Unknown provider '{provider}', falling back to hf")
            return _generate_hf(query, context)
    except Exception as exc:
        logger.exception("LLM provider '%s' failed: %s", provider, exc)
        print(f"[RAG] Provider '{provider}' failed: {exc}")

        # Auto-fallback: if OpenAI failed, try HF
        if provider == "openai":
            print("[RAG] Falling back to HF provider")
            try:
                return _generate_hf(query, context)
            except Exception as fallback_exc:
                logger.exception("HF fallback also failed: %s", fallback_exc)
                print(f"[RAG] HF fallback failed: {fallback_exc}")

        return ""
