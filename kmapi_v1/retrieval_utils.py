"""Embedding, retrieval, and query variant utilities separated from main.
"""
from typing import List, Dict, Tuple, Iterable
import time, requests, re, math
from logger import get_logger
from vectorstore import PGVectorRetriever
from config import LLM_API, NORMALIZE_EMBEDDINGS, VECTOR_K, DATABASE_URL, DOMAIN_KEYWORDS

log = get_logger("km_retrieval")

pg_retriever = None

def get_pg_retriever():
    global pg_retriever
    if pg_retriever is None:
        try:
            pg_retriever = PGVectorRetriever(DATABASE_URL)
        except Exception:
            log.exception("Failed to init PGVector retriever")
            pg_retriever = None
    return pg_retriever

COMMON_TYPO_MAP = {
    "reciept": "receipt",
    "recipt": "receipt",
    "paymnt": "payment",
    "recieved": "received",
    "receieved": "received",
    "adress": "address",
}

ACRONYM_EXPANSIONS = {
    "ecs": ["electronic clearing service"],
}

def apply_typo_corrections(text: str) -> str:
    lower = text.lower()
    for wrong, right in COMMON_TYPO_MAP.items():
        if wrong in lower:
            lower = lower.replace(wrong, right)
    return lower if text.islower() else lower

def expand_query_variants(original: str) -> List[str]:
    base = apply_typo_corrections(original)
    variants = [base]
    lower = base.lower()
    for acro, expansions in ACRONYM_EXPANSIONS.items():
        if acro in lower.split():
            for exp in expansions:
                if exp not in lower:
                    variants.append(exp)
                    variants.append(f"{base} {exp}")
    seen = set(); out = []
    for v in variants:
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def embed_query_via_api(text: str, timings: dict, trace_events, trace_id, t0, maybe_trace):
    emb = []
    try:
        t_emb = time.time(); r = requests.post(f"{LLM_API}/embed", json={"text": text}, timeout=15); r.raise_for_status(); emb = r.json().get("embedding", [])
        if NORMALIZE_EMBEDDINGS and emb:
            s = sum(e*e for e in emb) or 1.0; l2 = math.sqrt(s); emb = [e / l2 for e in emb]
        timings['embedding'] = time.time() - t_emb
        maybe_trace(trace_events, trace_id, t0, "embedded_query", dim=len(emb))
    except Exception:
        log.exception("Embedding API call failed")
    return emb


def retrieve_pgvector(query_embedding, k: int, timings: dict, trace_events, trace_id, t0, maybe_trace):
    contexts, citations, retrieval_meta = [], [], []
    retr = get_pg_retriever()
    if retr and query_embedding:
        try:
            t_r = time.time(); rows = retr.similarity_search(query_embedding, k=k); timings['retrieve'] = time.time() - t_r
            for content, score, source, vid in rows:
                contexts.append(content)
                citations.append({"source": source, "id": vid, "score": score})
                retrieval_meta.append({"score": score, "source": source, "id": vid})
            maybe_trace(trace_events, trace_id, t0, "retrieval_results", backend='pgvector', hits=len(rows), k=k)
        except Exception:
            log.exception("PGVector retrieval failed")
    return contexts, citations, retrieval_meta
