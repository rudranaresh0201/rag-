from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .deps import require_api_key
from ..core.logging import get_logger
from ..models.schemas import QueryRequest
from ..services.rebuild_service import is_rebuild_locked
from ..services.retrieval_service import retrieve_chunks_sync, generate_answer_sync

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query")
async def query_endpoint(
    request: QueryRequest,
    _: None = None,
) -> dict[str, Any]:
    try:
        logger.info("[QUERY] /query HIT")

        if is_rebuild_locked():
            raise HTTPException(status_code=503, detail="System rebuilding index")

        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        logger.info("[QUERY] %s", query)

        result = await asyncio.to_thread(
            retrieve_chunks_sync,
            query,
            request.top_k,
            request.document_id,
        )

        context = str(result.get("context", "") or "")
        chunks = result.get("chunks", [])

        if not context.strip():
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "sources": [],
            }

        answer = await asyncio.to_thread(generate_answer_sync, query, context)

        return {
            "answer": answer,
            "sources": chunks,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[ERROR] /query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")
