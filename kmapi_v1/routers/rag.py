"""RAG-related endpoints (process and rag_simple)."""
from fastapi import APIRouter
import time, uuid as _uuid, requests
from typing import Dict, Any

from model import KMRequest
from config import PIPELINE_CTX, log, VECTOR_K, LLM_API
from pipeline import (
    safety_pre_check,
    classify_intent,
    rephrase_query,
    retrieve_and_build_prompt,
    generate_answer,
    post_safety_and_ground,
)
from retrieval_utils import (
    embed_query_via_api,
    expand_query_variants,
    retrieve_pgvector,
)

router = APIRouter()

# Local trace utility (mirrors original behavior)

def _maybe_trace(trace_events, trace_id: str, start_time: float, event: str, **data):
    if trace_events is not None:
        trace_events.append({"t": round(time.time() - start_time, 4), "event": event, **data})
    log.info("TRACE[%s] %s %s", trace_id, event, data if data else "")

@router.post("/process")
def process(req: KMRequest):
    t0 = time.time(); trace_enabled = bool(getattr(req, 'trace', False) or PIPELINE_CTX.trace_default)
    trace_id = str(_uuid.uuid4()); trace_events = [] if trace_enabled else None
    _maybe_trace(trace_events, trace_id, t0, "request_received", query_len=len(req.user_input))

    timings: Dict[str, float] = {}

    safety_result, early = safety_pre_check(PIPELINE_CTX, log, req.user_input, timings, trace_events, trace_id, t0)
    if early:
        if trace_events: early.update({"trace": trace_events, "trace_id": trace_id})
        return early

    intent_obj, early = classify_intent(PIPELINE_CTX, log, req.user_input, timings, trace_events, trace_id, t0)
    if early:
        if trace_events: early.update({"trace": trace_events, "trace_id": trace_id})
        return early

    rephrased = rephrase_query(PIPELINE_CTX, log, req.user_input, timings, trace_events, trace_id, t0)

    prompt, citations, contexts, scores, early = retrieve_and_build_prompt(
        PIPELINE_CTX, log,
        rephrased=rephrased,
        original_query=req.user_input,
        timings=timings,
        trace_events=trace_events,
        trace_id=trace_id,
        t0=t0,
        expand_variants_fn=expand_query_variants,
        embed_fn=lambda text, timings, trace_events, trace_id, t0: embed_query_via_api(text, timings, trace_events, trace_id, t0, _maybe_trace),
        multi_retrieve_fn=lambda emb, k, timings, trace_events, trace_id, t0: retrieve_pgvector(emb, k, timings, trace_events, trace_id, t0, _maybe_trace),
    )
    if early:
        if trace_events: early.update({"trace": trace_events, "trace_id": trace_id})
        return early
    if not prompt:
        return {"bot_output": "Retrieval failed. Try again later.", "citations": []}

    answer, early = generate_answer(PIPELINE_CTX, log, prompt, timings, trace_events, trace_id, t0)
    if early:
        if trace_events: early.update({"trace": trace_events, "trace_id": trace_id})
        return early
    if answer is None:
        return {"bot_output": "Synthesis failed. Try again later.", "citations": []}

    final_answer, post_obj, original_answer = post_safety_and_ground(PIPELINE_CTX, log, answer, contexts, timings, trace_events, trace_id, t0)

    total = time.time() - t0; timings['total'] = total; log.info("TIMINGS %s", timings)
    resp: Dict[str, Any] = {
        "bot_output": final_answer,
        "citations": citations,
        "timings": timings,
        "intent": intent_obj,
        "retrieval_stats": {"scores_present": any(isinstance(s,(int,float)) for s in scores), "backend": "pgvector"}
    }
    if 'post_inspect' in timings and post_obj is not None:
        resp['moderation_post'] = post_obj
    if PIPELINE_CTX.include_debug_answer and final_answer != original_answer:
        resp['debug_raw_answer'] = original_answer
    if trace_events:
        resp['trace_id'] = trace_id; resp['trace'] = trace_events; _maybe_trace(trace_events, trace_id, t0, "response_ready", answer_chars=len(final_answer))
    return resp

@router.post("/rag_simple")
def rag_simple(req: KMRequest):
    t0 = time.time(); trace_enabled = bool(getattr(req, 'trace', False)); trace_id = str(_uuid.uuid4())
    query = req.user_input; k = PIPELINE_CTX.vector_k; timings: Dict[str, float] = {}
    trace_events = [] if trace_enabled else None
    def trace(event, **data):
        if trace_events is not None:
            trace_events.append({"t": round(time.time()-t0,4), "event": event, **data})
        log.info("TRACE[%s] %s %s", trace_id, event, data if data else "")
    # Embed
    vector_emb = []
    try:
        emb_t0 = time.time(); r = requests.post(f"{LLM_API}/embed", json={"text": query}, timeout=15); r.raise_for_status(); vector_emb = r.json().get('embedding', []); timings['embedding'] = time.time() - emb_t0
        trace("embedded_query", dim=len(vector_emb))
    except Exception:
        log.exception("Embedding failed in rag_simple")
        return {"bot_output": "Embedding failed", "citations": []}
    # Retrieve
    contexts = []; citations = []; retr = retrieve_pgvector(vector_emb, k, timings, trace_events, trace_id, t0, trace)
    # retrieve_pgvector returns triple; adapt
    if isinstance(contexts, list) and not contexts:
        pass
    contexts, citations, _ = retr
    if not contexts:
        return {"bot_output": "I don't know.", "citations": [], "backend": 'pgvector'}
    context_block = "\n---\n".join(contexts); trace("assembled_context", total_chars=len(context_block))
    prompt = (
        "You are a concise assistant. Use ONLY the context. If answer absent, say I don't know.\n"
        f"Context:\n---\n{context_block}\n---\nQuestion: {query}\nAnswer:" )
    try:
        g_t0 = time.time(); r = requests.post(f"{LLM_API}/chat", json={"prompt": prompt}, timeout=40); timings['llm_generate'] = time.time() - g_t0; r.raise_for_status(); answer = r.json().get('answer','').strip(); trace("llm_answer", answer_chars=len(answer))
    except Exception:
        log.exception("LLM call failed in rag_simple")
        return {"bot_output": "Generation failed", "citations": citations, "backend": 'pgvector'}
    timings['total'] = time.time() - t0; resp = {"bot_output": answer, "citations": citations, "timings": timings, "backend": 'pgvector'}
    if trace_events is not None: resp['trace_id'] = trace_id; resp['trace'] = trace_events
    return resp
