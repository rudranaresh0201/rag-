from __future__ import annotations

import os
from typing import Any
from pathlib import Path
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .db import get_all_records, get_collection, reset_database
from .ingestion import ingest_pdf_file_path, InvalidPDFError, MissingDependencyError
from .retrieval import retrieve_chunks
from .llm_router import generate_answer

app = FastAPI(title="PDF RAG Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TEMP: allow all (dev fix)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("\n[CRASH] Unhandled exception:")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
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
    original_name = (file.filename or "").strip()
    print(f"[UPLOAD] Received filename: {original_name}")

    if not original_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    safe_name = Path(original_name).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    project_root = Path(__file__).resolve().parent.parent
    docs_dir = project_root / "docs"
    print(f"[UPLOAD] Resolved docs directory: {docs_dir}")
    import uuid

    unique_name = f"{Path(safe_name).stem}_{uuid.uuid4().hex}.pdf"
    save_path = docs_dir / unique_name
    print(f"[UPLOAD] Final save path: {save_path}")

    try:
        docs_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(status_code=500, detail="Permission denied creating docs folder.")
    except Exception as e:
        print("[UPLOAD ERROR]", repr(e))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    bytes_written = 0
    try:
        with save_path.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                bytes_written += len(chunk)

        print(f"[UPLOAD] Uploaded size (bytes): {bytes_written}")
        print(f"[UPLOAD] Save complete: {save_path}")

        if bytes_written <= 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_exists = save_path.exists()
        print(f"[UPLOAD] File exists check: {file_exists}")
        if not file_exists:
            raise HTTPException(status_code=400, detail="File write failed")

        actual_size = int(save_path.stat().st_size)
        print(f"[UPLOAD] File size after save: {actual_size}")
        if actual_size <= 0:
            raise HTTPException(status_code=400, detail="File write failed")

        print(f"[UPLOAD] Verified file exists: {save_path.exists()} size={actual_size}")
        print(f"[UPLOAD] Files in docs: {[p.name for p in docs_dir.glob('*') if p.is_file()]}")

        try:
            collection = get_collection()
            collection.delete(where={"file": safe_name})
            print(f"[UPLOAD] Removed existing chunks for filename: {safe_name}")
        except Exception as e:
            print("[UPLOAD ERROR]", repr(e))
            raise HTTPException(status_code=500, detail=f"Failed to deduplicate existing file: {str(e)}")

        try:
            result = ingest_pdf_file_path(str(save_path), safe_name, actual_size)
        except Exception as e:
            print("[INGEST ERROR]", repr(e))
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

        return result

    except PermissionError:
        raise HTTPException(status_code=500, detail="Permission denied while saving uploaded file.")
    except OSError as e:
        print("[UPLOAD ERROR]", repr(e))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except InvalidPDFError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MissingDependencyError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print("[UPLOAD ERROR]", repr(e))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents() -> dict[str, list[dict[str, Any]]]:
    records = get_all_records()
    metadatas = records.get("metadatas") or []

    docs: dict[str, dict[str, Any]] = {}

    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue

        filename = str(metadata.get("file", "")).strip()
        if not filename:
            continue

        uploaded_at = str(metadata.get("uploaded_at", ""))
        doc_id = str(metadata.get("doc_id", "")).strip()
        size = int(metadata.get("size", 0) or 0)

        if filename not in docs:
            docs[filename] = {
                "doc_id": doc_id,
                "filename": filename,
                "size": size,
                "uploaded_at": uploaded_at,
                "chunks": 0,
            }

        else:
            current_uploaded_at = str(docs[filename].get("uploaded_at", ""))
            if uploaded_at > current_uploaded_at:
                docs[filename]["uploaded_at"] = uploaded_at
                docs[filename]["doc_id"] = doc_id
                docs[filename]["size"] = size

        docs[filename]["chunks"] += 1

    return {"documents": list(docs.values())}


@app.post("/query")
def query_endpoint(request: QueryRequest) -> dict[str, Any]:
    try:
        print("\n[DEBUG] /query HIT")

        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        result = retrieve_chunks(
            query=query,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        print("[DEBUG] retrieval done")

        context = str(result.get("context", "") or "")
        chunks = result.get("chunks", [])

        print("[DEBUG] FORCING LLM CALL")

        answer = generate_answer(query, context)

        print("[DEBUG] RAW ANSWER:", str(answer)[:200])

        return {
            "answer": answer,
            "sources": chunks
        }

    except HTTPException:
        raise
    except Exception as e:
        print("\n[ERROR IN /query]:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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