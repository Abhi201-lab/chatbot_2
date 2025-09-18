"""Configuration, environment loading, logger, and shared context objects.
"""
from typing import Iterable
from environment import load_env, env
from logger import get_logger
from pipeline import PipelineContext

load_env()
log = get_logger("km_api")

LLM_API = env("LLM_API_URL")
INCLUDE_DEBUG_ANSWER = env("INCLUDE_DEBUG_ANSWER", default="0", required=False) in ("1", "true", "TRUE", "yes")
ENFORCE_GROUNDED = env("ENFORCE_GROUNDED", default="1", required=False) in ("1", "true", "TRUE", "yes")
GROUNDING_MIN_OVERLAP = float(env("GROUNDING_MIN_OVERLAP", default="0.07", required=False))
RETRIEVAL_MIN_SCORE = float(env("RETRIEVAL_MIN_SCORE", default="0.30", required=False))
VECTOR_K = int(env("VECTOR_K", default="4", required=False))
DATABASE_URL = env("DATABASE_URL")
TRACE_DEFAULT = env("TRACE_DEFAULT", default="0", required=False) in ("1","true","TRUE","yes")
NORMALIZE_EMBEDDINGS = env("NORMALIZE_EMBEDDINGS", default="1", required=False) in ("1","true","TRUE","yes")
DOMAIN_KEYWORDS = {
    "pay","payment","online","bill","rebate","receipt","ecs","deposit","tariff","meter",
    "load","reconnect","ac","air","conditioner","wiring","security","advance"
}

PIPELINE_CTX = PipelineContext(
    llm_api=LLM_API,
    vector_k=VECTOR_K,
    retrieval_min_score=RETRIEVAL_MIN_SCORE,
    grounding_min_overlap=GROUNDING_MIN_OVERLAP,
    enforce_grounded=ENFORCE_GROUNDED,
    include_debug_answer=INCLUDE_DEBUG_ANSWER,
    domain_keywords=DOMAIN_KEYWORDS,
    trace_default=TRACE_DEFAULT,
)
