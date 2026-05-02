from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request

from  core.config import get_allowed_origins
from core.logging import get_logger
from db import get_embedder
from retrieval import warmup_bm25_index
from services.rebuild_service import rebuild_from_r2_if_empty
from api.routes_core import router as core_router
from api.routes_query import router as query_router
from api.routes_documents import router as documents_router

app = FastAPI(title="PDF RAG Backend", version="2.0.0")
logger = get_logger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception):
    logger.exception("[ERROR] Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
def startup_warmup() -> None:
    try:
        get_embedder()

        def _run_rebuild() -> None:
            rebuild_from_r2_if_empty()
            warmup_bm25_index()

        threading.Thread(target=_run_rebuild, daemon=True).start()
    except Exception as exc:
        logger.exception("[ERROR] Startup warmup failed: %s", exc)


app.include_router(core_router)
app.include_router(query_router)
app.include_router(documents_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8003, reload=True)
