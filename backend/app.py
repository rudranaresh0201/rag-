from __future__ import annotations

import logging
from collections import OrderedDict
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .db import delete_document, get_all_records, reset_database
    from .ingestion import InvalidPDFError, MissingDependencyError, ingest_pdf_file_path
    from .llm import generate_answer
    from .retrieval import retrieve_chunks
except ImportError:
    from db import delete_document, get_all_records, reset_database
    from ingestion import InvalidPDFError, MissingDependencyError, ingest_pdf_file_path
    from llm import generate_answer
    from retrieval import retrieve_chunks

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=12)


class SourceItem(BaseModel):
    id: int
    text: str
    document: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


def _trim_to_complete_sentence(text: str) -> str:
    value = " ".join(str(text).split()).strip()
    if not value:
        return ""

    matches = list(re.finditer(r"[.!?]", value))
    if matches:
        last_end = matches[-1].end()
        value = value[:last_end].strip()

    return value


def _build_structured_answer(raw_output: str, query: str, sources: List[dict[str, Any]]) -> str:
    fallback = "Information extracted from provided context."
    text = str(raw_output or "").replace("\r", "").strip()

    summary = ""
    explanation = ""
    key_points: list[str] = []

    summary_match = re.search(r"(?is)summary\s*:\s*(.*?)(?:\n\s*key points\s*:|$)", text)
    points_match = re.search(r"(?is)key points\s*:\s*(.*?)(?:\n\s*explanation\s*:|$)", text)
    explanation_match = re.search(r"(?is)explanation\s*:\s*(.*?)(?:\n\s*sources\s*:|$)", text)

    if summary_match:
        summary = _trim_to_complete_sentence(summary_match.group(1))
    if explanation_match:
        explanation = _trim_to_complete_sentence(explanation_match.group(1))

    if points_match:
        for line in points_match.group(1).split("\n"):
            cleaned = line.strip().lstrip("-*").strip()
            if cleaned:
                key_points.append(_trim_to_complete_sentence(cleaned) or cleaned)

    if not summary:
        sentence_candidates = re.split(r"(?<=[.!?])\s+", _trim_to_complete_sentence(text))
        summary = sentence_candidates[0].strip() if sentence_candidates and sentence_candidates[0].strip() else ""

    if not explanation:
        explanation = _trim_to_complete_sentence(text)

    if not key_points:
        sentence_candidates = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+", text)
            if len(segment.strip()) > 20
        ]
        key_points = [
            _trim_to_complete_sentence(segment) or segment
            for segment in sentence_candidates[:3]
        ]

    summary = summary or fallback
    explanation = explanation or fallback
    key_points = [point for point in key_points if point]
    if not key_points:
        key_points = [fallback]

    sources_list = "\n".join(
        f"- [{source['id']}] {source['document']}"
        + (f" (Page {source['page']})" if source.get("page") else "")
        for source in sources
    )

    return (
        "Summary:\n"
        f"{summary}\n\n"
        "Key Points:\n"
        + "\n".join(f"- {point}" for point in key_points[:5])
        + "\n\n"
        "Explanation:\n"
        f"{explanation}\n\n"
        "Sources:\n"
        f"{sources_list}"
    )


def _extract_from_context_fallback(context: str, query: str) -> str:
    normalized = " ".join(str(context).split()).strip()
    if not normalized:
        return ""

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]
    if not sentences:
        return normalized[:1200]

    query_terms = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(token) > 2
    }

    ranked_sentences: list[tuple[int, str]] = []
    for sentence in sentences:
        sentence_terms = {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", sentence.lower())
            if len(token) > 2
        }
        overlap = len(query_terms & sentence_terms)
        ranked_sentences.append((overlap, sentence))

    ranked_sentences.sort(key=lambda item: item[0], reverse=True)
    extracted = [item[1] for item in ranked_sentences if item[0] > 0][:4]
    if not extracted:
        extracted = sentences[:4]

    summary = extracted[0]
    key_points = extracted[:3]
    explanation = " ".join(extracted)

    return (
        "Summary:\n"
        f"{summary}\n\n"
        "Key Points:\n"
        + "\n".join(f"- {item}" for item in key_points)
        + "\n\n"
        "Explanation:\n"
        f"{explanation}"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.info("Upload rejected filename=%s status=rejected reason=invalid_file", file.filename)
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_file",
                "message": "Only PDF files are allowed.",
            },
        )

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE:
                logger.info(
                    "Upload rejected filename=%s size=%s status=rejected reason=file_too_large",
                    file.filename,
                    content_length,
                )
                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "file_too_large",
                        "message": "File exceeds 200 MB limit",
                    },
                )
        except ValueError:
            logger.warning("Invalid content-length header for filename=%s", file.filename)

    chunk_size = 1024 * 1024  # 1MB
    size = 0
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_path = tmp_file.name
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    logger.info(
                        "Upload rejected filename=%s size=%s status=rejected reason=file_too_large",
                        file.filename,
                        size,
                    )
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "file_too_large",
                            "message": "File exceeds 200 MB limit",
                        },
                    )
                tmp_file.write(chunk)

        if size == 0:
            logger.info("Upload rejected filename=%s status=rejected reason=empty_file", file.filename)
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_file",
                    "message": "Uploaded file is empty.",
                },
            )

        await file.seek(0)
        logger.info("Upload received filename=%s size=%s status=accepted", file.filename, size)

        result = ingest_pdf_file_path(
            pdf_path=temp_path,
            filename=file.filename,
            file_size=size,
        )
    except MissingDependencyError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "missing_dependency",
                "message": str(exc),
            },
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "parsing_failure",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "parsing_failure",
                "message": "Failed to ingest PDF.",
            },
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temporary upload file: %s", temp_path)

    logger.info(
        "Upload processed filename=%s size=%s chunks=%s doc_id=%s",
        file.filename,
        size,
        result["chunks"],
        result["doc_id"],
    )

    return {
        "status": "success",
        "chunks": result["chunks"],
        "doc_id": result["doc_id"],
        "metadata": {
            "id": result["doc_id"],
            "name": result["filename"],
            "size": result["size"],
            "uploaded_at": result["uploaded_at"],
        },
    }


@app.get("/documents")
def list_documents() -> List[dict[str, Any]]:
    records = get_all_records()
    metadatas = records.get("metadatas") or []

    by_doc_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for metadata in metadatas:
        if not metadata:
            continue

        doc_id = str(metadata.get("doc_id", ""))
        if not doc_id:
            continue

        if doc_id not in by_doc_id:
            by_doc_id[doc_id] = {
                "id": doc_id,
                "name": str(metadata.get("file", "unknown.pdf")),
                "size": int(metadata.get("size", 0)),
                "uploaded_at": str(metadata.get("uploaded_at", "")),
                "chunks": 0,
            }

        by_doc_id[doc_id]["chunks"] += 1

    documents = list(by_doc_id.values())
    documents.sort(key=lambda item: item.get("uploaded_at", ""), reverse=True)
    return documents


@app.delete("/document/{id}")
def delete_single_document(id: str) -> dict[str, str]:
    if not id.strip():
        raise HTTPException(status_code=400, detail="id is required.")

    try:
        delete_document(id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to delete document.") from exc

    return {"status": "success"}


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        no_info_answer = "Insufficient information in provided documents."
        query = request.query

        print("[RAG] Query received:", query)

        retrieval = retrieve_chunks(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
        retrieval_context = str(retrieval.get("context", "") or "")

        raw_chunks = retrieval.get("chunks", [])
        if not raw_chunks:
            return {
                "answer": no_info_answer,
                "sources": [],
            }
        seen_chunk_texts: set[str] = set()
        top_chunks: list[dict[str, Any]] = []
        max_chunks = max(3, min(request.top_k, 6))

        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            raw_text = str(chunk.get("text", ""))
            normalized = " ".join(raw_text.split()).strip().lower()
            if not normalized or normalized in seen_chunk_texts:
                continue
            seen_chunk_texts.add(normalized)
            top_chunks.append(chunk)
            if len(top_chunks) >= max_chunks:
                break

        sources = []
        context_chunks: list[str] = []

        for chunk in top_chunks:
            if not isinstance(chunk, dict):
                continue

            raw_text = str(chunk.get("text", ""))
            cleaned_text = " ".join(raw_text.split()).strip()[:1600]
            if not cleaned_text:
                continue

            source_id = len(sources) + 1

            source_text = cleaned_text[:260].strip()
            if len(cleaned_text) > 260:
                source_text += "..."

            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}

            file_name = str(chunk.get("file", "")).strip() or str(metadata.get("file", "")).strip()
            document_name = Path(file_name).name if file_name else "Uploaded Doc"

            page = None
            page_val = chunk.get("page")
            if page_val is None:
                page_val = metadata.get("page")
            if isinstance(page_val, int):
                page = page_val
            elif isinstance(page_val, str) and page_val.isdigit():
                page = int(page_val)

            sources.append(
                {
                    "id": source_id,
                    "text": source_text,
                    "document": document_name,
                    "page": page,
                }
            )
            source_label = f"{document_name} (Page {page})" if page is not None else document_name
            context_chunks.append(f"[Source: {source_label}]\n{cleaned_text}")

        context_from_chunks = "\n\n".join(context_chunks)
        # Prefer bounded app-built context blocks to avoid oversized prompts.
        context = context_from_chunks.strip() if context_from_chunks.strip() else retrieval_context.strip()

        print("[RAG] Context length:", len(context))
        print("[RAG] Context sample:", context[:200])

        if not sources or not context:
            print("[RAG] Empty context after build; skipping LLM")
            return {
                "answer": no_info_answer,
                "sources": [],
            }

        output = generate_answer(query, context)
        output = output.strip()
        if not output or len(output) < 40:
            print("[RAG] Weak model output; using context extraction fallback")
            output = _extract_from_context_fallback(context, query)
        if "Summary:" not in output:
            output = _trim_to_complete_sentence(output)

        final_answer = _build_structured_answer(output, query, sources)

        print("[RAG] Answer generated successfully")
        print("[RAG] Sources:", sources)

        return {
            "answer": final_answer,
            "sources": sources,
        }

    except Exception as e:
        print("[RAG] Query error:", str(e))
        return {
            "answer": "Internal server error",
            "sources": [],
        }


@app.delete("/reset")
def reset() -> dict[str, str]:
    try:
        reset_database()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to reset vector database.") from exc

    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8003, reload=True)
