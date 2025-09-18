"""Knowledge Manager API entrypoint.

This file now acts as a lightweight application factory wiring together:
 - config (environment, logger, pipeline context)
 - routers (diagnostics, rag)
 - middleware (request logging)
 - retrieval utilities (imported indirectly by routers)

Business logic and helpers were extracted to:
 - pipeline.py (pipeline steps)
 - retrieval_utils.py (embedding + retrieval helpers)
 - routers/diagnostics.py (health, stats, debug retrieval)
 - routers/rag.py (process + rag_simple endpoints)
 - config.py (env + constants + context)

This modular structure improves testability and separation of concerns.
"""

from fastapi import FastAPI, Request
import time
from config import log
from routers.diagnostics import router as diagnostics_router
from routers.rag import router as rag_router


app = FastAPI(title="Knowledge Manager API", version="1.1.0")
app.include_router(diagnostics_router)
app.include_router(rag_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info(f"Incoming request: {request.method} {request.url}")
    try:
        resp = await call_next(request)
        log.info(f"Request completed: {request.method} {request.url} status={resp.status_code}")
        return resp
    except Exception:
        log.exception("Unhandled exception in request pipeline")
        return {"bot_output": "Internal server error", "citations": []}
    # Routers supply endpoints; nothing else needed here.



