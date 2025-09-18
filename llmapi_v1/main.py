from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
import json

from environment import load_env, env
from logger import get_logger
from model import TextRequest, EmbedRequest, SynthRequest, ChatRequest

# prompt loader import
from prompts.prompt_loader import format_prompt, PromptNotFoundError


load_env()
log = get_logger("llm_api")

client = OpenAI(api_key=env("OPENAI_API_KEY"))
SAFETY_DISABLE_BLOCK = env("SAFETY_DISABLE_BLOCK", default="0", required=False) in ("1","true","TRUE","yes","on")

app = FastAPI(title="LLM API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info(f"Incoming request: {request.method} {request.url}")
    try:
        resp = await call_next(request)
        log.info(f"Request completed: {request.method} {request.url} status={resp.status_code}")
        return resp
    except Exception:
        log.exception("Unhandled exception in request pipeline")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


def _chat_once(system: str | None, user: str, temperature: float = 0.2) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


@app.post("/inspect")
def inspect(req: TextRequest):
    text = req.text or ""
    hard_categories = {"illicit instructions", "self-harm", "sexual exploitation"}
    mild_categories = {"violence", "hate"}

    def sanitize(cats, text):
        """Remove implausible hard categories lacking supporting keywords."""
        if not cats:
            return cats, []
        txt = text.lower()
        hard_keyword_map = {
            "illicit instructions": ["bomb","explosive","weapon","hack","gun","poison"],
            "self-harm": ["suicide","self harm","kill myself","harm myself","self-harm"],
            "sexual exploitation": ["child","underage","exploit","sexual","minor"],
        }
        removed = []
        kept = []
        for c in cats:
            c_l = c.lower().strip()
            if c_l in hard_keyword_map:
                kw_list = hard_keyword_map[c_l]
                if not any(k in txt for k in kw_list):
                    removed.append(c_l)
                    continue
            kept.append(c_l)
        return kept, removed

    def decide(cats, raw_text, removed_hard=None):
        cats_norm = [c.lower().strip() for c in cats if isinstance(c, str)]
        hard_found = any(c in hard_categories for c in cats_norm)
        mild_found = any(c in mild_categories for c in cats_norm)
        block = hard_found
        policy = ("hard_category" if block else ("mild_categories" if mild_found else "clean"))
        if removed_hard:
            # If we removed all hard categories, recompute block
            if block and not any(c in hard_categories for c in cats_norm):
                block = False
                policy = "sanitized_hard_removed"
        if SAFETY_DISABLE_BLOCK and block:
            # Override block for debugging but annotate policy
            policy = policy + "_disabled_flag"
            block = False
        return {
            "safe": not block,
            "categories": cats_norm,
            "block": block,
            "policy": policy,
            **({"removed_hard_categories": removed_hard} if removed_hard else {}),
            **({"blocking_disabled": True} if SAFETY_DISABLE_BLOCK else {}),
        }

    # Try model-based safety classification using template
    try:
        tmpl = format_prompt("safety_inspect_v1", text=text)
        raw = _chat_once(tmpl['system'], tmpl['user'], temperature=0)
        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    pass
        if isinstance(data, dict) and 'categories' in data:
            cats = data.get('categories') or []
            if not isinstance(cats, list):
                cats = []
            sanitized, removed = sanitize(cats, text)
            return decide(sanitized, text, removed_hard=removed if removed else None)
        else:
            log.warning("Model safety response missing keys; falling back keyword check raw=%s", raw[:160])
    except PromptNotFoundError:
        log.warning("Safety template not found; using keyword fallback")
    except Exception:
        log.exception("Safety model classification failed; using keyword fallback")

    # Fallback keyword heuristic classification
    txt = text.lower()
    cats = []
    if any(k in txt for k in ["bomb recipe", "how to build a bomb", "make a bomb", "explosive"]):
        cats.append("illicit instructions")
    if any(k in txt for k in ["self harm", "suicide"]):
        cats.append("self-harm")
    if any(k in txt for k in ["kill", "murder"]):
        if "illicit instructions" not in cats:
            cats.append("violence")
    if any(k in txt for k in ["hate crime", "racist"]):
        cats.append("hate")
    sanitized, removed = sanitize(cats, text)
    return decide(sanitized, text, removed_hard=removed if removed else None)


@app.post("/embed")
def embed(req: EmbedRequest):
    try:
        emb = client.embeddings.create(model="text-embedding-ada-002", input=req.text)
        return {"embedding": emb.data[0].embedding}
    except Exception:
        log.exception("Embedding error")
        return JSONResponse(status_code=502, content={"detail": "Embedding provider error"})


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    prompt = f"{req.context}\n\nTask: {req.query}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return {"answer": resp.choices[0].message.content}
    except Exception:
        log.exception("Synthesize failed")
        return JSONResponse(status_code=502, content={"detail": "Synthesize provider error"})


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": req.prompt}],
            temperature=0.2,
        )
        return {"answer": resp.choices[0].message.content}
    except Exception:
        log.exception("Chat failed")
        return JSONResponse(status_code=502, content={"detail": "Chat provider error"})
