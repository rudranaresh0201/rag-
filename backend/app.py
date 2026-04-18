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
        no_info_answer = "No relevant information found in the provided document."
        query = request.query

        def normalize_answer(text: str) -> str:
            normalized = str(text).strip()

            if "Answer:" in normalized:
                normalized = normalized.split("Answer:")[-1]

            if "-" not in normalized:
                sentences = re.split(r"(?<=[.!?]) +", normalized)
                bullets = []
                for sentence in sentences[:4]:
                    sentence = sentence.strip()
                    if len(sentence) > 20:
                        bullets.append("- " + sentence)
                return "\n".join(bullets)

            return normalized

        print("QUERY RECEIVED:", query)

        retrieval = retrieve_chunks(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
        retrieval_context = retrieval.get("context", "")

        print("CONTEXT LENGTH:", len(retrieval_context))

        # FORCE SAFE EXECUTION
        if not retrieval_context or len(retrieval_context.strip()) == 0:
            print("EMPTY CONTEXT")
            return {
                "answer": no_info_answer,
                "sources": [],
            }

        print("CALLING GENERATE ANSWER")

        raw_chunks = retrieval.get("chunks", [])
        seen_chunk_texts: set[str] = set()
        top_chunks: list[dict[str, Any]] = []

        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            raw_text = str(chunk.get("text", ""))
            normalized = " ".join(raw_text.split()).strip().lower()
            if not normalized or normalized in seen_chunk_texts:
                continue
            seen_chunk_texts.add(normalized)
            top_chunks.append(chunk)
            if len(top_chunks) >= 3:
                break

        sources = []
        context_chunks: list[str] = []

        for chunk in top_chunks:
            if not isinstance(chunk, dict):
                continue

            raw_text = str(chunk.get("text", ""))
            cleaned_text = raw_text.replace("\n", " ").strip()[:700]
            if not cleaned_text:
                continue

            source_id = len(sources) + 1

            source_text = cleaned_text[:200].strip()
            if len(cleaned_text) > 200:
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
            context_chunks.append(f"[{source_id}] {cleaned_text}")

        context = "\n\n".join(context_chunks[:3])

        if not sources or not context.strip():
            return {
                "answer": no_info_answer,
                "sources": [],
            }

        output = generate_answer(query, context)
        output = normalize_answer(output)

        print("FINAL ANSWER:", output)

        if len(output.strip()) < 20:
            return {
                "answer": no_info_answer,
                "sources": sources,
            }

        sources_list = "\n".join(
            f"[{source['id']}] {source['document']}"
            + (f" • Page {source['page']}" if source.get("page") else "")
            for source in sources
        )
        final_answer = output + "\n\nSources:\n" + sources_list

        print("SENDING TO FRONTEND:", final_answer)

        return {
            "answer": final_answer,
            "sources": sources,
        }

    except Exception as e:
        print("QUERY ERROR:", str(e))
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
