"""Knowledge Manager API
"""

import sys
from fastapi import FastAPI, Request
import time, requests, re

from environment import load_env, env
from logger import get_logger
from model import KMRequest
from vectorstore import PGVectorRetriever
import uuid as _uuid

from prompts.prompt_loader import format_prompt, PromptNotFoundError


load_env()
log = get_logger("km_api")

LLM_API = env("LLM_API_URL")
INCLUDE_DEBUG_ANSWER = env("INCLUDE_DEBUG_ANSWER", default="0", required=False) in ("1", "true", "TRUE", "yes")
ENFORCE_GROUNDED = env("ENFORCE_GROUNDED", default="1", required=False) in ("1", "true", "TRUE", "yes")
GROUNDING_MIN_OVERLAP = float(env("GROUNDING_MIN_OVERLAP", default="0.07", required=False))

RETRIEVAL_MIN_SCORE = float(env("RETRIEVAL_MIN_SCORE", default="0.30", required=False))
DOMAIN_KEYWORDS = {
    "pay","payment","online","bill","rebate","receipt","ecs","deposit","tariff","meter",
    "load","reconnect","ac","air","conditioner","wiring","security","advance"
}
VECTOR_K = int(env("VECTOR_K", default="4", required=False))
DATABASE_URL = env("DATABASE_URL")  # now required
TRACE_DEFAULT = env("TRACE_DEFAULT", default="0", required=False) in ("1","true","TRUE","yes")
NORMALIZE_EMBEDDINGS = env("NORMALIZE_EMBEDDINGS", default="1", required=False) in ("1","true","TRUE","yes")

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


app = FastAPI(title="Knowledge Manager API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/vector_stats")
def vector_stats():
    """Basic stats for pgvector table (row count)."""
    retr = get_pg_retriever()
    if retr is None:
        return {"status": "error", "reason": "retriever not available"}
    # lightweight count
    try:
        from sqlalchemy import create_engine, text
        eng = retr.engine
        with eng.connect() as conn:
            count = conn.execute(text("SELECT count(*) FROM vector_chunks")).scalar() or 0
        return {"status": "ok", "row_count": int(count)}
    except Exception:
        log.exception("Failed to fetch vector stats")
        return {"status": "error", "reason": "stats_failed"}


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


def _call_llm_synthesize(query: str, context: str, timeout_sec: int = 15) -> str:
    """Wrapper for synthesize LLM endpoint (robust to failures)."""
    try:
        resp = requests.post(
            f"{LLM_API}/synthesize",
            json={"query": query, "context": context},
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception:
        log.exception("LLM synthesize call failed")
        return ""


def _maybe_trace(trace_events, trace_id: str, start_time: float, event: str, **data):
    """Append a structured trace event and mirror to logs."""
    if trace_events is not None:
        trace_events.append({
            "t": round(time.time() - start_time, 4),
            "event": event,
            **data
        })
    log.info("TRACE[%s] %s %s", trace_id, event, data if data else "")


def _embed_query_via_api(text: str, timings: dict, trace_events, trace_id, t0):
    """Call central embedding endpoint and return vector list[float]."""
    emb = []
    try:
        t_emb = time.time()
        r = requests.post(f"{LLM_API}/embed", json={"text": text}, timeout=15)
        r.raise_for_status()
        emb = r.json().get("embedding", [])
        if NORMALIZE_EMBEDDINGS and emb:
            import math
            s = sum(e*e for e in emb) or 1.0
            l2 = math.sqrt(s)
            emb = [e / l2 for e in emb]
        timings['embedding'] = time.time() - t_emb
        _maybe_trace(trace_events, trace_id, t0, "embedded_query", dim=len(emb))
    except Exception:
        log.exception("Embedding API call failed")
    return emb


def _retrieve_pgvector(query_embedding, k: int, timings: dict, trace_events, trace_id, t0):
    contexts, citations, retrieval_meta = [], [], []
    retr = get_pg_retriever()
    if retr and query_embedding:
        try:
            t_r = time.time(); rows = retr.similarity_search(query_embedding, k=k); timings['retrieve'] = time.time() - t_r
            for content, score, source, vid in rows:
                contexts.append(content)
                citations.append({"source": source, "id": vid, "score": score})
                retrieval_meta.append({"score": score, "source": source, "id": vid})
            _maybe_trace(trace_events, trace_id, t0, "retrieval_results", backend='pgvector', hits=len(rows), k=k)
        except Exception:
            log.exception("PGVector retrieval failed")
    return contexts, citations, retrieval_meta


# Simple domain typo corrections to improve recall (before embedding)
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
    # Add domain-specific acronyms here.
}

def _expand_query_variants(original: str) -> list[str]:
    base = _apply_typo_corrections(original)
    variants = [base]
    lower = base.lower()
    for acro, expansions in ACRONYM_EXPANSIONS.items():
        if acro in lower.split():  # crude token check
            for exp in expansions:
                if exp not in lower:
                    variants.append(exp)
                    variants.append(f"{base} {exp}")
    # Deduplicate preserving order
    seen = set(); out = []
    for v in variants:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def _apply_typo_corrections(text: str) -> str:
    lower = text.lower()
    for wrong, right in COMMON_TYPO_MAP.items():
        if wrong in lower:
            lower = lower.replace(wrong, right)
    return lower if text.islower() else lower  


@app.get("/debug_retrieve")
def debug_retrieve(q: str, k: int = 4):
    """Diagnostic endpoint: shows raw retrieval rows (content preview + score)."""
    t0 = time.time()
    q_fixed = _apply_typo_corrections(q)
    emb = _embed_query_via_api(q_fixed, {}, None, "debug", t0)
    if not emb:
        return {"query": q, "fixed": q_fixed, "error": "embed_failed"}
    contexts, citations, meta = _retrieve_pgvector(emb, k, {}, None, "debug", t0)
    previews = [c[:160].replace('\n',' ') for c in contexts]
    return {"query": q, "fixed": q_fixed, "hits": len(contexts), "previews": previews, "citations": citations}


@app.post("/process")
def process(req: KMRequest):
    """Full RAG pipeline with safety, intent, rephrase, retrieval, synthesis, post-safety & grounding.

    Optional tracing: pass {"trace": true} to receive an ordered list of internal events.
    """
    t0 = time.time()
    trace_enabled = bool(getattr(req, 'trace', False) or TRACE_DEFAULT)
    trace_id = str(_uuid.uuid4())
    trace_events = [] if trace_enabled else None
    _maybe_trace(trace_events, trace_id, t0, "request_received", query_len=len(req.user_input))

    timings = {}
    # 1) safety inspect
    try:
        t_inspect_start = time.time(); safety_result = requests.post(f"{LLM_API}/inspect", json={"text": req.user_input}, timeout=8).json()
        timings['safety_inspect'] = time.time() - t_inspect_start
        _maybe_trace(trace_events, trace_id, t0, "safety_pre", **{k: safety_result.get(k) for k in ("policy","block","categories")})
        if safety_result.get("block", (not safety_result.get("safe", True))):
            log.warning("Request blocked by safety check policy=%s categories=%s", safety_result.get('policy'), safety_result.get('categories'))
            return {"bot_output": "The query was blocked by safety checks.", "citations": [], "moderation": safety_result, **({"trace": trace_events, "trace_id": trace_id} if trace_events else {})}
    except requests.exceptions.RequestException:
        log.exception("Inspection call failed")
        return {"bot_output": "Inspection failed. Try again later.", "citations": []}

    # 2) intent classification (in-domain vs out_of_scope) BEFORE rephrase & retrieval
    intent_obj = {"intent": "unknown", "out_of_scope": True, "reason": "classifier_failed"}
    try:
        t_intent_start = time.time()
        try:
            intent_tmpl = format_prompt("intent_classify_v1", question=req.user_input)
            sys_part = intent_tmpl.get('system') or ''
            user_part = intent_tmpl.get('user') or ''
            intent_prompt = f"{sys_part}\n\n{user_part}" if sys_part else user_part
        except PromptNotFoundError:
            intent_prompt = (
                "You output ONLY minified JSON: {\"intent\":string,\"out_of_scope\":boolean,\"reason\":string}. "
                "If query not about billing, online/advance payment, ECS, security deposit, tariff/billing, name/address change, reconnection, load enhancement, AC installation, meter, wiring say out_of_scope true.\nQuery: "
                + req.user_input
            )
        log.info("POST %s/chat (intent)", LLM_API)
        ic = requests.post(f"{LLM_API}/chat", json={"prompt": intent_prompt}, timeout=12)
        ic.raise_for_status()
        raw_ic = ic.json().get('answer', '')
        import json as _json, re as _re
        parsed = None
        try:
            parsed = _json.loads(raw_ic)
        except Exception:
            m = _re.search(r"\{.*\}", raw_ic, _re.DOTALL)
            if m:
                try: parsed = _json.loads(m.group(0))
                except Exception: pass
        if isinstance(parsed, dict):
            intent_obj['intent'] = parsed.get('intent', intent_obj['intent'])
            intent_obj['out_of_scope'] = bool(parsed.get('out_of_scope', intent_obj['out_of_scope']))
            intent_obj['reason'] = parsed.get('reason', intent_obj['reason'])
        timings['intent_classify'] = time.time() - t_intent_start
        log.info("intent=%s out_of_scope=%s reason=%s", intent_obj['intent'], intent_obj['out_of_scope'], intent_obj['reason'])
        _maybe_trace(trace_events, trace_id, t0, "intent_result", **intent_obj)
        if intent_obj.get('out_of_scope', True):
            resp = {
                "bot_output": "I don't have information about that. Please ask about billing, payment, ECS, security deposit, tariff, meter, reconnection, load, AC installation, wiring or related support topics.",
                "citations": [],
                "intent": intent_obj,
                "timings": timings
            }
            if trace_events: resp.update({"trace": trace_events, "trace_id": trace_id})
            return resp
    except Exception:
        log.exception("Intent classification failed; proceeding without gating")

    # 3) rephrase (best-effort) to improve retrieval via prompt template (optionally could condition on intent)
    try:
        t_rephrase_start = time.time()
        try:
            tmpl = format_prompt("rephrase_v1", question=req.user_input)
            synth_query = tmpl['user']
            synth_context = tmpl['system']
        except (PromptNotFoundError, KeyError) as e:
            log.warning("Falling back to inline rephrase due to template issue: %s", e)
            synth_query = 'Return STRICT JSON only, no prose: {"intent": string, "rephrased": string suitable for vector search}.'
            synth_context = f"User query: {req.user_input}"
        raw = _call_llm_synthesize(query=synth_query, context=synth_context)
        rephrased = req.user_input
        if raw:
            import json as _json
            try:
                obj = _json.loads(raw)
                rephrased = obj.get("rephrased") or obj.get("intent") or req.user_input
            except Exception:
                if len(raw) < 160:
                    rephrased = raw
        log.info("rephrased query: '%s'", rephrased[:160].replace("\n", " "))
        timings['rephrase'] = time.time() - t_rephrase_start
        _maybe_trace(trace_events, trace_id, t0, "rephrase_result", rephrased=rephrased[:160])
    except Exception:
        log.warning("Rephrase step failed; using original query")
        rephrased = req.user_input

    # 3) retrieve (multi-pass with expansions if first pass empty)
    try:
        t_retrieve_start = time.time()
        variants = _expand_query_variants(rephrased)
        all_rows = []  # (content, score, source, vid)
        variant_meta = []  # track which variant produced hits
        for vi, variant in enumerate(variants):
            query_embedding = _embed_query_via_api(variant, timings, trace_events, trace_id, t0)
            if not query_embedding:
                continue
            contexts_v, citations_v, retrieval_meta_v = _retrieve_pgvector(query_embedding, VECTOR_K, timings, trace_events, trace_id, t0)
            if contexts_v:
                for c, cit in zip(contexts_v, citations_v):
                    all_rows.append((c, cit))
                variant_meta.append({"variant": variant, "hits": len(contexts_v)})
            # If we already have enough hits, we can break early to save latency
            if len(all_rows) >= VECTOR_K:
                break
        # Merge by (source,id,score) uniqueness keeping insertion order
        merged_contexts = []
        merged_citations = []
        seen_keys = set()
        for content, cit in all_rows:
            key = (cit['source'], cit['id'])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_contexts.append(content)
            merged_citations.append(cit)
            if len(merged_contexts) >= VECTOR_K:
                break
        contexts, citations = merged_contexts, merged_citations
        # Synthesize retrieval_meta (scores only) for stats
        retrieval_meta = [{"score": c.get('score'), "source": c.get('source'), "id": c.get('id')} for c in citations]
        timings['retrieval_total'] = time.time() - t_retrieve_start
        scores = [m['score'] for m in retrieval_meta if isinstance(m.get('score'), (int,float))]
        if not contexts:
            _maybe_trace(trace_events, trace_id, t0, "retrieval_empty", variants=variants)
            return {"bot_output": "Sorry, I couldn’t find relevant information in the documents.", "citations": [], "expansions_tried": variants}
        # Log sample hits
        for i, meta in enumerate(retrieval_meta[:VECTOR_K]):
            log.info("hit[%d] backend=pgvector score=%s source=%s", i, None if meta.get('score') is None else round(meta['score'],4), meta.get('source'))
        # Retrieval confidence gating
        try:
            usable_scores = scores
            best = max(usable_scores) if usable_scores else None
            avg = sum(usable_scores)/len(usable_scores) if usable_scores else None
            combined_len = sum(len(c) for c in contexts)
            # Gate only if both poor similarity and tiny context and no domain keywords visible
            context_join_lower = " ".join(contexts).lower()
            has_domain_token = any(k in context_join_lower for k in DOMAIN_KEYWORDS)
            if best is not None and best < RETRIEVAL_MIN_SCORE and combined_len < 120 and not has_domain_token:
                log.warning("Low confidence retrieval (best=%.3f len=%d domain=%s) -> unknown", best, combined_len, has_domain_token)
                return {"bot_output": "I don't know.", "citations": citations, "retrieval": {"best_score": best, "threshold": RETRIEVAL_MIN_SCORE, "context_length": combined_len}}
            _maybe_trace(trace_events, trace_id, t0, "similarity_stats", best=best, avg=avg, context_chars=combined_len, backend='pgvector')
        except Exception:
            log.warning("Retrieval confidence gating failed (continuing)")
        # build final answer prompt via template
        try:
            answer_tmpl = format_prompt("rag_answer_v1", context="\n---\n".join(contexts), user_query=req.user_input)
            prompt = f"{answer_tmpl['system']}\n\n{answer_tmpl['user']}"
        except Exception as e:
            log.warning("Falling back to inline answer prompt: %s", e)
            prompt = (
                "Answer ONLY from the context. If not present, say you don't know.\n"
                "Include brief citations (source/section if present).\n\n"
                f"Context:\n---\n{chr(10).join(contexts)}\n---\n\nQ: {req.user_input}\nA:" )
    except Exception:
        log.exception("Retrieval error")
        return {"bot_output": "Retrieval failed. Try again later.", "citations": []}

    # 4) LLM synthesis
    try:
        t_llm_start = time.time()
        log.info("POST %s/chat", LLM_API)
        log.info("prompt length=%d", len(prompt))
        r = requests.post(f"{LLM_API}/chat", json={"prompt": prompt}, timeout=40)
        r.raise_for_status()
        answer = r.json().get("answer", "").strip()
        timings['llm_generate'] = time.time() - t_llm_start
        _maybe_trace(trace_events, trace_id, t0, "llm_answer", chars=len(answer))
    except requests.exceptions.RequestException:
        log.exception("LLM synthesis failed")
        return {"bot_output": "Synthesis failed. Try again later.", "citations": []}

    original_answer = answer

    # 5) safety post-check FIRST on original answer
    post = None
    try:
        post_t0 = time.time(); post = requests.post(f"{LLM_API}/inspect", json={"text": answer}, timeout=8).json(); timings['post_inspect'] = time.time() - post_t0
        block_gen = post.get("block", (not post.get("safe", True)))
        _maybe_trace(trace_events, trace_id, t0, "safety_post", block=block_gen, policy=post.get('policy'), categories=post.get('categories'))
        if block_gen:
            log.warning("Generated answer failed post-inspection policy=%s categories=%s", post.get('policy'), post.get('categories'))
            answer = "The generated content was moderated and withheld."
    except Exception:
        log.warning("Post-inspection failed")

    # 6) grounding enforcement after safety (only if answer not replaced by moderation message)
    try:
        if ENFORCE_GROUNDED and not answer.startswith("The generated content was moderated"):
            joined_context = "\n".join(contexts).lower()
            tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", original_answer.lower())
            domain_in_answer = any(k in original_answer.lower() for k in DOMAIN_KEYWORDS)
            if tokens:
                overlapping = [tok for tok in tokens if tok in joined_context]
                overlap_ratio = (len(overlapping) / max(len(tokens), 1)) if tokens else 0.0
                log.info("Grounding heuristic tokens=%d overlap=%d ratio=%.2f domain=%s", len(tokens), len(overlapping), overlap_ratio, domain_in_answer)
                short_answer = len(original_answer.strip()) < 40
                if not domain_in_answer and overlap_ratio < GROUNDING_MIN_OVERLAP and not short_answer:
                    log.warning("Answer ungrounded (ratio=%.2f<thr=%.2f) -> replacing with I don't know", overlap_ratio, GROUNDING_MIN_OVERLAP)
                    answer = "I don't know."
                _maybe_trace(trace_events, trace_id, t0, "grounding", ratio=round(overlap_ratio,3), domain_in_answer=domain_in_answer)
            else:
                log.info("Grounding skipped (no tokens)")
    except Exception:
        log.warning("Grounding enforcement failed (continuing with current answer)")

    total = time.time() - t0
    timings['total'] = total
    log.info("TIMINGS %s", timings)
    resp = {"bot_output": answer, "citations": citations, "timings": timings, "intent": intent_obj, "retrieval_stats": {"scores_present": any(isinstance(s,(int,float)) for s in scores), "backend": "pgvector"}}
    # Always surface post moderation details if we performed it
    try:
        if 'post_inspect' in timings:
            resp['moderation_post'] = post  # type: ignore
    except Exception:
        pass
    if INCLUDE_DEBUG_ANSWER and answer != original_answer:
        resp['debug_raw_answer'] = original_answer
    if trace_events:
        resp['trace_id'] = trace_id
        resp['trace'] = trace_events
        _maybe_trace(trace_events, trace_id, t0, "response_ready", answer_chars=len(answer))
    return resp


@app.post("/rag_simple")
def rag_simple(req: KMRequest):

    t0 = time.time()
    trace_enabled = bool(getattr(req, 'trace', False))
    trace_id = str(_uuid.uuid4())
    query = req.user_input
    k = VECTOR_K
    timings = {}
    trace_events = [] if trace_enabled else None
    def trace(event, **data):
        if trace_events is not None:
            trace_events.append({"t": round(time.time()-t0,4), "event": event, **data})
        log.info("TRACE[%s] %s %s", trace_id, event, data if data else "")
    # 1. Embed
    vector_emb = []
    try:
        emb_t0 = time.time(); r = requests.post(f"{LLM_API}/embed", json={"text": query}, timeout=15); r.raise_for_status(); vector_emb = r.json().get('embedding', []); timings['embedding'] = time.time() - emb_t0
        trace("embedded_query", dim=len(vector_emb))
    except Exception:
        log.exception("Embedding failed in rag_simple")
        return {"bot_output": "Embedding failed", "citations": []}

    # 2. Retrieve
    contexts = []
    citations = []
    used_backend = 'pgvector'
    retr = get_pg_retriever()
    if retr:
        r_t0 = time.time(); rows = retr.similarity_search(vector_emb, k=k); timings['retrieve'] = time.time() - r_t0
        trace("retrieval_results", backend="pgvector", hits=len(rows), k=k)
        for content, score, source, vid in rows:
            contexts.append(content)
            citations.append({"source": source, "id": vid, "score": score})
    else:
        return {"bot_output": "Retriever unavailable", "citations": []}

    if not contexts:
        return {"bot_output": "I don't know.", "citations": [], "backend": used_backend}

    # 3. Simple prompt
    context_block = "\n---\n".join(contexts)
    trace("assembled_context", total_chars=len(context_block))
    prompt = (
        "You are a concise assistant. Use ONLY the context. If answer absent, say I don't know.\n"\
        f"Context:\n---\n{context_block}\n---\nQuestion: {query}\nAnswer:" )

    # 4. LLM generate
    try:
        g_t0 = time.time(); r = requests.post(f"{LLM_API}/chat", json={"prompt": prompt}, timeout=40); timings['llm_generate'] = time.time() - g_t0
        r.raise_for_status(); answer = r.json().get('answer','').strip()
        trace("llm_answer", answer_chars=len(answer))
    except Exception:
        log.exception("LLM call failed in rag_simple")
        return {"bot_output": "Generation failed", "citations": citations, "backend": used_backend}

    timings['total'] = time.time() - t0
    resp = {"bot_output": answer, "citations": citations, "timings": timings, "backend": used_backend}
    if trace_events is not None:
        resp['trace_id'] = trace_id
        resp['trace'] = trace_events
    return resp


