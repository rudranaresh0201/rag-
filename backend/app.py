from __future__ import annotations

import os
import tempfile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .db import get_all_records, reset_database
from .ingestion import ingest_pdf_file_path, InvalidPDFError, MissingDependencyError
from .retrieval import retrieve_chunks
from .llm_router import generate_answer

app = FastAPI(title="PDF RAG Backend", version="2.0.0")

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


# ---------------- ROUTES ---------------- #

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "uploaded.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    tmp_path = ""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = ingest_pdf_file_path(tmp_path, filename, len(contents))
        return result

    except InvalidPDFError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except MissingDependencyError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Upload failed")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/documents")
def list_documents() -> list[dict[str, Any]]:
    records = get_all_records()
    metadatas = records.get("metadatas") or []
    return [m for m in metadatas if isinstance(m, dict)]


@app.post("/query")
def query_endpoint(request: QueryRequest) -> dict[str, Any]:
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        result = retrieve_chunks(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        context = str(result.get("context", "") or "")
        chunks = result.get("chunks", [])

        print(f"[RAG] Context length before LLM: {len(context)}")
        print("[FIXED] Context length sent to LLM:", len(context))
        print(f"[RAG] Context preview: {context[:300]}")

        valid_chunks = [chunk for chunk in chunks if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()]
        if not valid_chunks and context.strip():
            valid_chunks = [{"text": context.strip(), "file": "unknown", "page": None}]

        def summarize_chunks(chunk_items: list[dict[str, Any]]) -> str:
            lines: list[str] = []
            for idx, chunk in enumerate(chunk_items[:3], start=1):
                text = str(chunk.get("text", "")).strip()
                if text:
                    lines.append(f"{idx}. {text[:260]}")
            if lines:
                return "Closest relevant information:\n" + "\n".join(lines)
            return "No relevant context found in retrieved chunks."

        answer = generate_answer(query, context)
        print("[DEBUG] LLM CALLED")
        print("[DEBUG] Answer after LLM:", str(answer)[:200])

        if not str(answer or "").strip():
            answer = summarize_chunks(valid_chunks)

        sources = []
        for idx, chunk in enumerate(valid_chunks, start=1):
            sources.append(
                {
                    "id": idx,
                    "text": chunk.get("text", "")[:300],
                    "document": chunk.get("file", "unknown"),
                    "page": chunk.get("page"),
                }
            )

        if not sources and context.strip():
            sources = [{
                "id": 1,
                "text": context[:300],
                "document": "unknown",
                "page": None,
            }]

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception:
        raise HTTPException(status_code=500, detail="Query failed")


@app.delete("/reset")
def reset() -> dict[str, str]:
    try:
        reset_database()
        return {"status": "success"}
    except Exception:
        raise HTTPException(status_code=500, detail="Reset failed")


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8003, reload=True)