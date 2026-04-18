from __future__ import annotations

import re
from typing import List


def clean_text(text: str) -> str:
    """Normalize whitespace and remove obvious non-content artifacts."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 650, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping token-like chunks.
    We approximate tokens by whitespace-delimited words for predictable performance.
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    step = chunk_size - overlap
    chunks: List[str] = []

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

    return chunks
