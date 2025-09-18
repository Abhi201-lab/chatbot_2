"""Diagnostics & health related endpoints.
"""
from fastapi import APIRouter
import time
from config import VECTOR_K
from retrieval_utils import get_pg_retriever, apply_typo_corrections, embed_query_via_api, retrieve_pgvector
from config import log

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/vector_stats")
def vector_stats():
    retr = get_pg_retriever()
    if retr is None:
        return {"status": "error", "reason": "retriever not available"}
    try:
        from sqlalchemy import text
        eng = retr.engine
        with eng.connect() as conn:
            count = conn.execute(text("SELECT count(*) FROM vector_chunks")).scalar() or 0
        return {"status": "ok", "row_count": int(count)}
    except Exception:
        log.exception("Failed to fetch vector stats")
        return {"status": "error", "reason": "stats_failed"}

@router.get("/debug_retrieve")
def debug_retrieve(q: str, k: int = 4):
    t0 = time.time()
    q_fixed = apply_typo_corrections(q)
    emb = embed_query_via_api(q_fixed, {}, None, "debug", t0, lambda *a, **kw: None)
    if not emb:
        return {"query": q, "fixed": q_fixed, "error": "embed_failed"}
    contexts, citations, _ = retrieve_pgvector(emb, k, {}, None, "debug", t0, lambda *a, **kw: None)
    previews = [c[:160].replace('\n',' ') for c in contexts]
    return {"query": q, "fixed": q_fixed, "hits": len(contexts), "previews": previews, "citations": citations}
