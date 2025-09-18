
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



