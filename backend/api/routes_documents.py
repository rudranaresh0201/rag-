from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from deps import require_api_key
from core.logging import get_logger
from db import get_all_records, get_collection
from storage import list_all_pdfs_in_r2, delete_pdf_from_r2

router = APIRouter()
logger = get_logger(__name__)


@router.get("/documents")
def list_documents(_: None = Depends(require_api_key)) -> dict[str, list[dict[str, Any]]]:
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
                "status": "ok",
            }
        else:
            current_uploaded_at = str(docs[filename].get("uploaded_at", ""))
            if uploaded_at > current_uploaded_at:
                docs[filename]["uploaded_at"] = uploaded_at
                docs[filename]["doc_id"] = doc_id
                docs[filename]["size"] = size

        docs[filename]["chunks"] += 1

    doc_ids = {str(doc.get("doc_id", "")).strip() for doc in docs.values() if doc.get("doc_id")}
    try:
        r2_keys = list_all_pdfs_in_r2()
    except Exception as exc:
        logger.warning("[DOCUMENTS] R2 list failed: %s", exc)
        r2_keys = []

    for key in r2_keys:
        if not key:
            continue
        doc_id = key.split("/", 1)[0] if "/" in key else ""
        if doc_id and doc_id in doc_ids:
            continue
        filename = Path(key).name or "document.pdf"
        corrupt_key = f"{doc_id}:{filename}"
        if corrupt_key in docs:
            continue
        docs[corrupt_key] = {
            "doc_id": doc_id,
            "filename": filename,
            "size": 0,
            "uploaded_at": "",
            "chunks": 0,
            "status": "corrupt",
        }

    return {"documents": list(docs.values())}


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, _: None = Depends(require_api_key)) -> dict[str, str]:
    try:
        collection = get_collection()
        records = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        metadatas = records.get("metadatas") or []
        s3_key = None
        for metadata in metadatas:
            if isinstance(metadata, dict) and metadata.get("s3_key"):
                s3_key = str(metadata.get("s3_key"))
                break

        if s3_key:
            try:
                delete_pdf_from_r2(s3_key)
            except Exception as exc:
                logger.warning("[DELETE] Failed to delete from R2: %s", exc)

        collection.delete(where={"doc_id": doc_id})
        return {"status": "success"}
    except Exception as exc:
        logger.exception("[ERROR] Delete failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")
