from __future__ import annotations

from datetime import datetime, timezone
import uuid
from pathlib import Path
from typing import TypedDict

try:
    import fitz
except ImportError:
    fitz = None

from db import embed_texts, get_collection
from utils import chunk_text, clean_text

from core.logging import get_logger

logger = get_logger(__name__)


class InvalidPDFError(Exception):
    pass


class MissingDependencyError(Exception):
    pass


class IngestionResult(TypedDict):
    chunks: int
    doc_id: str
    filename: str
    size: int
    uploaded_at: str


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if fitz is None:
        raise MissingDependencyError("PyMuPDF not installed. Run: pip install pymupdf")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
    except Exception as exc:
        raise InvalidPDFError("Invalid or unreadable PDF.") from exc

    text = "\n".join(pages)
    return clean_text(text)


def extract_text_from_pdf_path(pdf_path: str) -> str:
    if fitz is None:
        raise MissingDependencyError("PyMuPDF not installed. Run: pip install pymupdf")

    try:
        with fitz.open(pdf_path) as doc:
            pages = [page.get_text("text") for page in doc]
    except Exception as exc:
        raise InvalidPDFError("Invalid or unreadable PDF.") from exc

    text = "\n".join(pages)
    return clean_text(text)


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    file_size: int,
    doc_id: str | None = None,
    s3_key: str | None = None,
    file_hash: str | None = None,
) -> IngestionResult:
    logger.info("Starting PDF ingestion filename=%s size=%s", filename, file_size)
    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        raise InvalidPDFError("PDF has no extractable text.")

    chunks = chunk_text(text=text, chunk_size=650, overlap=100)
    chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not chunks:
        raise InvalidPDFError("No valid text chunks were produced.")

    embeddings = embed_texts(chunks)
    doc_id = str(doc_id or uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat()

    ids = [f"{Path(filename).stem}-{uuid.uuid4()}" for _ in chunks]
    metadatas = [
        {
            "file": filename,
            "doc_id": doc_id,
            "size": int(file_size),
            "uploaded_at": uploaded_at,
            "chunk_index": index,
            "s3_key": s3_key,
            "content_hash": file_hash or "",
        }
        for index, _ in enumerate(chunks)
    ]

    collection = get_collection()
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    logger.info("CHUNKS STORED: %s", len(chunks))
    logger.info("TOTAL IN DB: %s", collection.count())
    logger.info(
        "Completed PDF ingestion filename=%s size=%s chunks=%s doc_id=%s",
        filename,
        file_size,
        len(chunks),
        doc_id,
    )

    return {
        "chunks": len(chunks),
        "doc_id": doc_id,
        "filename": filename,
        "size": int(file_size),
        "uploaded_at": uploaded_at,
    }


def ingest_pdf_file_path(
    pdf_path: str,
    filename: str,
    file_size: int,
    doc_id: str | None = None,
    s3_key: str | None = None,
    file_hash: str | None = None,
) -> IngestionResult:
    logger.info("Starting PDF ingestion filename=%s size=%s", filename, file_size)
    text = extract_text_from_pdf_path(pdf_path)
    if not text:
        raise InvalidPDFError("PDF has no extractable text.")

    chunks = chunk_text(text=text, chunk_size=650, overlap=100)
    chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not chunks:
        raise InvalidPDFError("No valid text chunks were produced.")

    embeddings = embed_texts(chunks)
    doc_id = str(doc_id or uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat()

    ids = [f"{Path(filename).stem}-{uuid.uuid4()}" for _ in chunks]
    metadatas = [
        {
            "file": filename,
            "doc_id": doc_id,
            "size": int(file_size),
            "uploaded_at": uploaded_at,
            "chunk_index": index,
            "s3_key": s3_key,
            "content_hash": file_hash or "",
        }
        for index, _ in enumerate(chunks)
    ]

    collection = get_collection()
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
    logger.info("CHUNKS STORED: %s", len(chunks))
    logger.info("TOTAL IN DB: %s", collection.count())
    logger.info(
        "Completed PDF ingestion filename=%s size=%s chunks=%s doc_id=%s",
        filename,
        file_size,
        len(chunks),
        doc_id,
    )

    return {
        "chunks": len(chunks),
        "doc_id": doc_id,
        "filename": filename,
        "size": int(file_size),
        "uploaded_at": uploaded_at,
    }


