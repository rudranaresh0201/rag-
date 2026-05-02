from __future__ import annotations

import tempfile
import threading
import uuid
from pathlib import Path

from core.logging import get_logger
from db import get_collection
from ingestion import ingest_pdf_file_path
from storage import list_all_pdfs_in_r2, download_pdf_from_r2
from utils.hashing import compute_hash

logger = get_logger(__name__)
_rebuilding = False
_rebuild_lock = threading.Lock()


def is_rebuilding() -> bool:
    return _rebuilding


def is_rebuild_locked() -> bool:
    return _rebuild_lock.locked()


def rebuild_from_r2_if_empty() -> None:
    global _rebuilding
    if not _rebuild_lock.acquire(blocking=False):
        return

    try:
        _rebuilding = True
        logger.info("[REBUILD] Start")
        collection = get_collection()
        if int(collection.count()) > 0:
            return

        try:
            keys = list_all_pdfs_in_r2()
        except Exception as exc:
            logger.warning("[REBUILD] R2 list failed: %s", exc)
            return

        if not keys:
            logger.info("[REBUILD] R2 bucket is empty; skipping rebuild")
            return

        for key in keys:
            tmp_path: Path | None = None
            try:
                filename = Path(key).name or "document.pdf"
                doc_id = key.split("/")[0] if "/" in key else str(uuid.uuid4())

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    tmp_path = Path(temp_file.name)

                download_pdf_from_r2(key, tmp_path)
                file_size = int(tmp_path.stat().st_size)
                content_hash = compute_hash(tmp_path.read_bytes())
                ingest_pdf_file_path(
                    str(tmp_path),
                    filename,
                    file_size,
                    doc_id=doc_id,
                    s3_key=key,
                    file_hash=content_hash,
                )
            except Exception as exc:
                logger.warning("[REBUILD] Failed for %s: %s", key, exc)
                continue
            finally:
                if tmp_path:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
    finally:
        _rebuilding = False
        _rebuild_lock.release()
        logger.info("[REBUILD] End")
