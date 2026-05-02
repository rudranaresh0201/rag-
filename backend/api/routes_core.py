from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from deps import require_api_key
from core.config import get_max_upload_bytes
from core.logging import get_logger
from db import get_collection, reset_database
from ingestion import InvalidPDFError, MissingDependencyError
from services.ingestion_service import run_ingest_task
from services.rebuild_service import is_rebuilding
from tasks import create_task, get_task_status

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
def health() -> dict[str, Any]:
    collection = get_collection()
    return {
        "status": "ok",
        "rebuilding": bool(is_rebuilding()),
        "chunks": int(collection.count()),
    }


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: None = None,
) -> dict[str, Any]:
    if is_rebuilding():
        raise HTTPException(
            status_code=503,
            detail="System rebuilding, try later",
        )

    original_name = (file.filename or "").strip()
    logger.info("[INGEST] Received filename: %s", original_name)

    if not original_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    safe_name = Path(original_name).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    project_root = Path(__file__).resolve().parent.parent
    docs_dir = project_root / "docs"

    unique_name = f"{Path(safe_name).stem}_{uuid.uuid4().hex}.pdf"
    save_path = docs_dir / unique_name

    try:
        docs_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
            raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as exc:
        logger.exception("[ERROR] Upload init failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")

    bytes_written = 0
    max_bytes = get_max_upload_bytes()
    try:
        with save_path.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    try:
                        out_file.flush()
                    except Exception:
                        pass
                    try:
                        save_path.unlink()
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "file_too_large",
                            "message": f"File exceeds {max_bytes} bytes.",
                        },
                    )

        if bytes_written <= 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_exists = save_path.exists()
        if not file_exists:
            raise HTTPException(status_code=400, detail="File write failed")

        actual_size = int(save_path.stat().st_size)
        if actual_size <= 0:
            raise HTTPException(status_code=400, detail="File write failed")

        task_id = uuid.uuid4().hex
        create_task(task_id)
        background_tasks.add_task(run_ingest_task, task_id, save_path, safe_name, actual_size)

        return {"task_id": task_id, "status": "pending"}

    except PermissionError:
        raise HTTPException(status_code=500, detail="Internal server error")
    except OSError as exc:
        logger.exception("[ERROR] Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")
    except InvalidPDFError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except MissingDependencyError as exc:
        raise HTTPException(status_code=500, detail="Internal server error")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[ERROR] Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/task/{task_id}")
def get_task_status_route(task_id: str, _: None = Depends(require_api_key)) -> dict[str, str]:
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status": str(task.get("status", "pending")),
        "error": str(task.get("error", "")),
    }


@router.delete("/reset")
def reset(confirm: bool = False, _: None = Depends(require_api_key)) -> dict[str, str]:
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="Reset requires confirm=true")
        reset_database()
        return {"status": "success"}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
