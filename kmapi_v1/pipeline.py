"""Pipeline helper functions for the /process endpoint.

These were extracted from the original monolithic implementation in main.py to
improve readability, testability, and reuse. All functions are intentionally
public (no leading underscore) per user request.

Contract notes:
- Each function returns primitive data plus an optional early-response dict when
  a terminal condition is reached. Caller should short‑circuit if early response
  is not None.
- All functions accept the timing dict and tracing parameters where relevant so
  observability is preserved without global state coupling.

Functions:
    safety_pre_check
    classify_intent
    rephrase_query
    retrieve_and_build_prompt
    generate_answer
    post_safety_and_ground

A small PipelineContext dataclass captures external dependencies (config flags,
constants, network base URLs) making unit tests simpler (can inject fakes).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
import time, requests, re
import uuid as _uuid

from prompts.prompt_loader import format_prompt, PromptNotFoundError  # type: ignore

# NOTE: We don't import logger or environment here; caller passes logger & config

@dataclass
class PipelineContext:
    llm_api: str
    vector_k: int
    retrieval_min_score: float
    grounding_min_overlap: float
    enforce_grounded: bool
    include_debug_answer: bool
    domain_keywords: Iterable[str]
    trace_default: bool = False

# Helper to add structured trace

def maybe_trace(trace_events, trace_id: str, start_time: float, event: str, logger, **data):
    if trace_events is not None:
        trace_events.append({
            "t": round(time.time() - start_time, 4),
            "event": event,
            **data
        })
    logger.info("TRACE[%s] %s %s", trace_id, event, data if data else "")


def safety_pre_check(ctx: PipelineContext, logger, text: str, timings: Dict[str, float], trace_events, trace_id: str, t0: float):
    try:
        t_inspect_start = time.time(); safety_result = requests.post(f"{ctx.llm_api}/inspect", json={"text": text}, timeout=8).json(); timings['safety_inspect'] = time.time() - t_inspect_start
        maybe_trace(trace_events, trace_id, t0, "safety_pre", logger, **{k: safety_result.get(k) for k in ("policy","block","categories")})
        if safety_result.get("block", (not safety_result.get("safe", True))):
            logger.warning("Request blocked by safety check policy=%s categories=%s", safety_result.get('policy'), safety_result.get('categories'))
            return safety_result, {"bot_output": "The query was blocked by safety checks.", "citations": [], "moderation": safety_result}
        return safety_result, None
    except requests.exceptions.RequestException:
        logger.exception("Inspection call failed")
        return None, {"bot_output": "Inspection failed. Try again later.", "citations": []}


def classify_intent(ctx: PipelineContext, logger, user_input: str, timings: Dict[str, float], trace_events, trace_id: str, t0: float):
    intent_obj: Dict[str, Any] = {"intent": "unknown", "out_of_scope": True, "reason": "classifier_failed"}
    try:
        t_intent_start = time.time()
        try:
            intent_tmpl = format_prompt("intent_classify_v1", question=user_input)
            sys_part = intent_tmpl.get('system') or ''
            user_part = intent_tmpl.get('user') or ''
            intent_prompt = f"{sys_part}\n\n{user_part}" if sys_part else user_part
        except PromptNotFoundError:
            intent_prompt = (
                "You output ONLY minified JSON: {\"intent\":string,\"out_of_scope\":boolean,\"reason\":string}. "
                "If query not about billing, online/advance payment, ECS, security deposit, tariff/billing, name/address change, reconnection, load enhancement, AC installation, meter, wiring say out_of_scope true.\nQuery: "
                + user_input
            )
        logger.info("POST %s/chat (intent)", ctx.llm_api)
        ic = requests.post(f"{ctx.llm_api}/chat", json={"prompt": intent_prompt}, timeout=12)
        ic.raise_for_status()
        raw_ic = ic.json().get('answer', '')
        import json as _json, re as _re
        parsed = None
        try:
            parsed = _json.loads(raw_ic)
        except Exception:
            m = _re.search(r"\{.*\}", raw_ic, _re.DOTALL)
            if m:
                try:
                    parsed = _json.loads(m.group(0))
                except Exception:
                    pass
        if isinstance(parsed, dict):
            intent_obj['intent'] = parsed.get('intent', intent_obj['intent'])
            intent_obj['out_of_scope'] = bool(parsed.get('out_of_scope', intent_obj['out_of_scope']))
            intent_obj['reason'] = parsed.get('reason', intent_obj['reason'])
        timings['intent_classify'] = time.time() - t_intent_start
        logger.info("intent=%s out_of_scope=%s reason=%s", intent_obj['intent'], intent_obj['out_of_scope'], intent_obj['reason'])
        maybe_trace(trace_events, trace_id, t0, "intent_result", logger, **intent_obj)
        if intent_obj.get('out_of_scope', True):
            return intent_obj, {
                "bot_output": "I don't have information about that. Please ask about billing, payment, ECS, security deposit, tariff, meter, reconnection, load, AC installation, wiring or related support topics.",
                "citations": [],
                "intent": intent_obj,
            }
    except Exception:
        logger.exception("Intent classification failed; proceeding without gating")
    return intent_obj, None


def rephrase_query(ctx: PipelineContext, logger, original_query: str, timings: Dict[str, float], trace_events, trace_id: str, t0: float):
    try:
        t_rephrase_start = time.time()
        try:
            tmpl = format_prompt("rephrase_v1", question=original_query)
            synth_query = tmpl['user']
            synth_context = tmpl['system']
        except (PromptNotFoundError, KeyError) as e:
            logger.warning("Falling back to inline rephrase due to template issue: %s", e)
            synth_query = 'Return STRICT JSON only, no prose: {"intent": string, "rephrased": string suitable for vector search}.'
            synth_context = f"User query: {original_query}"
        # Synthesize using /synthesize endpoint for robustness
        try:
            r = requests.post(f"{ctx.llm_api}/synthesize", json={"query": synth_query, "context": synth_context}, timeout=15)
            r.raise_for_status(); raw = r.json().get('answer', '')
        except Exception:
            raw = ''
        rephrased = original_query
        if raw:
            import json as _json
            try:
                obj = _json.loads(raw)
                rephrased = obj.get("rephrased") or obj.get("intent") or original_query
            except Exception:
                if len(raw) < 160:
                    rephrased = raw
        logger.info("rephrased query: '%s'", rephrased[:160].replace("\n", " "))
        timings['rephrase'] = time.time() - t_rephrase_start
        maybe_trace(trace_events, trace_id, t0, "rephrase_result", logger, rephrased=rephrased[:160])
        return rephrased
    except Exception:
        logger.warning("Rephrase step failed; using original query")
        return original_query

# retrieval utilities retained in main (embedding, typo expansion) are referenced from main; this module only orchestrates

def retrieve_and_build_prompt(ctx: PipelineContext, logger, *, rephrased: str, original_query: str, timings: Dict[str, float], trace_events, trace_id: str, t0: float, expand_variants_fn, embed_fn, multi_retrieve_fn):
    try:
        t_retrieve_start = time.time()
        variants = expand_variants_fn(rephrased)
        all_rows = []  # (content, citation_dict)
        for variant in variants:
            query_embedding = embed_fn(variant, timings, trace_events, trace_id, t0)
            if not query_embedding:
                continue
            contexts_v, citations_v, _ = multi_retrieve_fn(query_embedding, ctx.vector_k, timings, trace_events, trace_id, t0)
            if contexts_v:
                for c, cit in zip(contexts_v, citations_v):
                    all_rows.append((c, cit))
            if len(all_rows) >= ctx.vector_k:
                break
        seen_keys = set(); contexts: List[str] = []; citations: List[Dict[str, Any]] = []
        for content, cit in all_rows:
            key = (cit.get('source'), cit.get('id'))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            contexts.append(content)
            citations.append(cit)
            if len(contexts) >= ctx.vector_k:
                break
        timings['retrieval_total'] = time.time() - t_retrieve_start
        scores = [c.get('score') for c in citations if isinstance(c.get('score'), (int, float))]
        if not contexts:
            maybe_trace(trace_events, trace_id, t0, "retrieval_empty", logger, variants=variants)
            return None, [], [], [], {"bot_output": "Sorry, I couldn’t find relevant information in the documents.", "citations": [], "expansions_tried": variants}
        try:
            best = max(scores) if scores else None
            avg = sum(scores)/len(scores) if scores else None
            combined_len = sum(len(c) for c in contexts)
            context_join_lower = " ".join(contexts).lower()
            has_domain_token = any(k in context_join_lower for k in ctx.domain_keywords)
            if best is not None and best < ctx.retrieval_min_score and combined_len < 120 and not has_domain_token:
                logger.warning("Low confidence retrieval (best=%.3f len=%d domain=%s) -> unknown", best, combined_len, has_domain_token)
                return None, citations, contexts, scores, {"bot_output": "I don't know.", "citations": citations, "retrieval": {"best_score": best, "threshold": ctx.retrieval_min_score, "context_length": combined_len}}
            maybe_trace(trace_events, trace_id, t0, "similarity_stats", logger, best=best, avg=avg, context_chars=combined_len, backend='pgvector')
        except Exception:
            logger.warning("Retrieval confidence gating failed (continuing)")
        try:
            answer_tmpl = format_prompt("rag_answer_v1", context="\n---\n".join(contexts), user_query=original_query)
            prompt = f"{answer_tmpl['system']}\n\n{answer_tmpl['user']}"
        except Exception as e:
            logger.warning("Falling back to inline answer prompt: %s", e)
            prompt = (
                "Answer ONLY from the context. If not present, say you don't know.\n"\
                "Include brief citations (source/section if present).\n\n"\
                f"Context:\n---\n{'\n'.join(contexts)}\n---\n\nQ: {original_query}\nA:"
            )
        return prompt, citations, contexts, scores, None
    except Exception:
        logger.exception("Retrieval error")
        return None, [], [], [], {"bot_output": "Retrieval failed. Try again later.", "citations": []}


def generate_answer(ctx: PipelineContext, logger, prompt: str, timings: Dict[str, float], trace_events, trace_id: str, t0: float):
    try:
        t_llm_start = time.time(); logger.info("POST %s/chat", ctx.llm_api); logger.info("prompt length=%d", len(prompt))
        r = requests.post(f"{ctx.llm_api}/chat", json={"prompt": prompt}, timeout=40); r.raise_for_status(); answer = r.json().get("answer", "").strip(); timings['llm_generate'] = time.time() - t_llm_start
        maybe_trace(trace_events, trace_id, t0, "llm_answer", logger, chars=len(answer))
        return answer, None
    except requests.exceptions.RequestException:
        logger.exception("LLM synthesis failed")
        return None, {"bot_output": "Synthesis failed. Try again later.", "citations": []}


def post_safety_and_ground(ctx: PipelineContext, logger, answer: str, contexts: List[str], timings: Dict[str, float], trace_events, trace_id: str, t0: float):
    original_answer = answer
    post = None
    try:
        post_t0 = time.time(); post = requests.post(f"{ctx.llm_api}/inspect", json={"text": answer}, timeout=8).json(); timings['post_inspect'] = time.time() - post_t0
        block_gen = post.get("block", (not post.get("safe", True)))
        maybe_trace(trace_events, trace_id, t0, "safety_post", logger, block=block_gen, policy=post.get('policy'), categories=post.get('categories'))
        if block_gen:
            logger.warning("Generated answer failed post-inspection policy=%s categories=%s", post.get('policy'), post.get('categories'))
            answer = "The generated content was moderated and withheld."
    except Exception:
        logger.warning("Post-inspection failed")
    try:
        if ctx.enforce_grounded and not answer.startswith("The generated content was moderated"):
            joined_context = "\n".join(contexts).lower()
            tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", original_answer.lower())
            domain_in_answer = any(k in original_answer.lower() for k in ctx.domain_keywords)
            if tokens:
                overlapping = [tok for tok in tokens if tok in joined_context]
                overlap_ratio = (len(overlapping) / max(len(tokens), 1)) if tokens else 0.0
                logger.info("Grounding heuristic tokens=%d overlap=%d ratio=%.2f domain=%s", len(tokens), len(overlapping), overlap_ratio, domain_in_answer)
                short_answer = len(original_answer.strip()) < 40
                if not domain_in_answer and overlap_ratio < ctx.grounding_min_overlap and not short_answer:
                    logger.warning("Answer ungrounded (ratio=%.2f<thr=%.2f) -> replacing with I don't know", overlap_ratio, ctx.grounding_min_overlap)
                    answer = "I don't know."
                maybe_trace(trace_events, trace_id, t0, "grounding", logger, ratio=round(overlap_ratio,3), domain_in_answer=domain_in_answer)
            else:
                logger.info("Grounding skipped (no tokens)")
    except Exception:
        logger.warning("Grounding enforcement failed (continuing with current answer)")
    return answer, post, original_answer
