from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from core.logging import get_logger
from db import get_collection
from ingestion import ingest_pdf_file_path
from storage import upload_pdf_to_r2, build_r2_key
from tasks import set_task_status, set_task_error

logger = get_logger(__name__)


def run_ingest_task(task_id: str, save_path: Path, safe_name: str, actual_size: int) -> None:
    doc_id = uuid.uuid4().hex
    s3_key = build_r2_key(doc_id, safe_name)
    try:
        set_task_status(task_id, "processing")
        hasher = hashlib.sha256()
        with save_path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        content_hash = hasher.hexdigest()

        collection = get_collection()
        existing = collection.get(where={"content_hash": content_hash})
        if existing.get("ids"):
            logger.info("[INGEST] Duplicate content hash; skipping ingestion filename=%s", safe_name)
            set_task_status(task_id, "done")
            return

        collection.delete(where={"file": safe_name})
        ingest_pdf_file_path(
            str(save_path),
            safe_name,
            actual_size,
            doc_id=doc_id,
            s3_key=s3_key,
            file_hash=content_hash,
        )
        import os

        if os.getenv("USE_R2", "false").lower() == "true":
            from storage import maybe_upload_to_r2
            maybe_upload_to_r2(save_path, doc_id, safe_name)
        set_task_status(task_id, "done")
        logger.info("[INGEST] Success filename=%s doc_id=%s", safe_name, doc_id)
    except Exception:
        logger.exception("[INGEST] Failed filename=%s", safe_name)
        try:
            get_collection().delete(where={"doc_id": doc_id})
        except Exception:
            pass
        set_task_status(task_id, "failed")
        set_task_error(task_id, "Ingestion failed")
    finally:
        try:
            save_path.unlink()
        except OSError:
            pass
