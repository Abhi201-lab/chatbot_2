# Chatbot RAG Microservices Platform

## 1. Overview
A modular Retrieval‑Augmented Generation (RAG) system composed of independent FastAPI / Streamlit services. It ingests documents, embeds them into a Postgres + pgvector store, retrieves semantically relevant chunks, and synthesizes grounded answers with layered safety, intent gating, and tracing.

Core goals:
- Clear separation of concerns per service
- Replaceable LLM / embedding provider (currently OpenAI) via `llmapi_v1`
- Production‑oriented retrieval (pgvector cosine similarity)
- Observability (trace events, debug retrieval endpoint)
- Safety + grounding heuristics to mitigate hallucinations

## 2. Architecture (High Level)
```
+----------------+        +----------------+        +----------------+        +----------------+
|  chatui_v1     | --->   |  chat_api_v1   | --->   |  kmapi_v1      | --->   |  llmapi_v1      |
| (Streamlit UI) |  API   | (Gateway /     |  RAG   | (RAG Orchestr. |  LLM   | (LLM + Embed +  |
|                | <---   |  Session Mgmt) | <---   | Retrieval +    | <---  |  Safety Utils)  |
+----------------+        +----------------+        | Safety + Trace |        +----------------+
                                                      |    |
                                                      v    |
                                               +----------------+            
                                               | ingestapi_v1   | --
                                               | (Upload +      |   \
                                               |  Chunk + Embed)|    --> Postgres + pgvector (vector_chunks)
                                               +----------------+   /
                                                                    
```

## 3. Services
| Service | Port (container) | Role |
|---------|------------------|------|
| `chatui_v1` | 8501 | Streamlit chat front-end |
| `chat_api_v1` | 8000 | Gateway / orchestration entry point |
| `kmapi_v1` | 8001 | Knowledge Manager: intent, rephrase, multi‑pass retrieval, grounding, synthesis orchestration |
| `llmapi_v1` | 8002 | Embedding, chat, synthesis and safety inspection wrapper |
| `ingestapi_v1` | 8003 | Document upload & JSON ingest: chunking + embedding + persistence |
| `postgres_v1` | 5432 (host 5543) | Postgres with pgvector extension |
Note: Ports numbers can vary 

## 4. Data Flow (Full `/process` Pipeline)
1. Request enters `chat_api_v1` (optionally adds session/user metadata)from UI.
2. Forwarded to `kmapi_v1 /process`.
3. Pre‑safety inspection (moderation) via `llmapi_v1/inspect`.
4. Intent classification (in‑domain gating).
5. Rephrase query via prompt template (improves semantic recall).
6. Multi‑pass retrieval:
   - Typo correction (receipt, received, etc.)
   - Acronym expansion (e.g. ECS → Electronic Clearing Service)
   - Embedding via `llmapi_v1/embed` (OpenAI model currently)
   - `pgvector` cosine similarity search (`embedding <=> query_vec` → distance → similarity=1-distance)
7. Gating logic: heuristic low‑confidence fallback.
8. Answer synthesis using context (`llmapi_v1/chat`).
9. Post‑safety inspection (moderates generated answer).
10. Grounding heuristic: token overlap + domain keyword presence.
11. Trace events aggregated (if enabled) and returned.

`/rag_simple` endpoint : embed → retrieve → answer 

## 5. Retrieval & Embeddings
- Embeddings: OpenAI `text-embedding-ada-002` (1536 dims) via `llmapi_v1`.
- Storage: Postgres table `vector_chunks(id, source, content, embedding vector(1536), ...)`.
- Normalization: Optional L2 normalization at ingest & query (`NORMALIZE_EMBEDDINGS=1`).
- Distance: pgvector cosine operator `<=>` (returns distance). Similarity computed as `1 - distance` in SQL select.
- Multi-pass retrieval variants: base corrected + acronym expansions + fused phrase.
- Debug endpoint: `GET /debug_retrieve?q=...&k=...` (shows raw hits, previews, scores).

## 6. Recent Retrieval Fix (Important)
Original parameter binding produced `operator does not exist: vector <=> numeric[]` errors (driver bound parameter as numeric array). Fix: embed vector literal directly (`[f1,f2,...]`) in SQL query for `kmapi_v1` retrieval. This restored hits. (A future refinement could register a proper psycopg2 adapter.)

## 7. Grounding & Safety
- Safety: Pre and post answer via `llmapi_v1/inspect` (returns categories + block flag). Env flag can disable blocking but still annotate.
- Grounding heuristic: Overlap ratio of answer tokens with concatenated contexts + domain keyword detection. If below `GROUNDING_MIN_OVERLAP` and not domain relevant, answer replaced with "I don't know." (Disabled during tuning with `ENFORCE_GROUNDED=0`).

## 8. Environment Variables (Key)
| Variable | Purpose | Typical |
|----------|---------|---------|
| `DATABASE_URL` | Postgres DSN (with pgvector) | postgresql+psycopg2://... |
| `LLM_API_URL` | Base URL for llmapi service | http://llmapi_v1:8002 |
| `VECTOR_K` | Retrieval top-K | 4–8 |
| `RETRIEVAL_MIN_SCORE` | Low-confidence gate threshold | 0.15–0.35 |
| `ENFORCE_GROUNDED` | Enable grounding enforcement | 1 / 0 |
| `GROUNDING_MIN_OVERLAP` | Token overlap threshold | 0.05–0.10 |
| `TRACE_DEFAULT` | Always return trace events | 0 / 1 |
| `INCLUDE_DEBUG_ANSWER` | Include raw answer pre-moderation | 0 / 1 |
| `NORMALIZE_EMBEDDINGS` | L2 normalize embeddings | 1 |
| `SAFETY_DISABLE_BLOCK` | Allow unsafe but annotate | 0 / 1 |

Legacy removed: `VECTOR_BACKEND`, `VECTOR_DB_PATH` (FAISS eliminated).

## 9. Running Locally (Docker Compose)
```powershell
# Build & start
docker compose up -d --build

# Tail logs (example)
docker compose logs -f kmapi_v1

# Stop all
docker compose down
```

Open services:
- UI: http://localhost:8502 
- Chat API: http://localhost:8080
- KM API debug: http://localhost:8081/debug_retrieve?q=test
- LLM API: http://localhost:8082/health
- Ingest API: http://localhost:8083/health
- Postgres: localhost:5543

## 10. Ingesting Documents
### File Upload (PDF/DOCX)
`POST /upload` (multipart) to `ingestapi_v1` with fields:
- file
- thread_id
- uploaded_by (optional)

### JSON Ingest
`POST /ingest` body:
```json
{
  "thread_id": "thread123",
  "chunk_size": 700,
  "chunk_overlap": 100,
  "docs": [ { "text": "Some content", "source": "guide.txt" } ]
}
```

Check counts:
```powershell
curl http://localhost:8083/vector_stats
```

## 11. Querying


```powershell
curl -X POST http://localhost:8081/rag_simple -H "Content-Type: application/json" -d '{"user_input":"how to pay bill","trace":true}'
```
Full pipeline:
```powershell
curl -X POST http://localhost:8081/process -H "Content-Type: application/json" -d '{"user_input":"payment receipt not received","trace":true}'
```

## 12. Tracing
Returned `trace` array events (timestamps relative to request start):
- request_received
- safety_pre
- intent_result
- rephrase_result
- retrieval_results
- similarity_stats
- llm_answer
- safety_post
- grounding
- response_ready

## 13. Normalization & Indexing (WIP)
Normalization script (if needed for legacy rows):
```powershell
$env:DATABASE_URL="postgresql://postgres:123@localhost:5543/chatdb"
python scripts/normalize_vectors.py --dry-run
python scripts/normalize_vectors.py
```
Optional indexes:
```sql
CREATE INDEX IF NOT EXISTS vector_chunks_embedding_ivfflat_cos
  ON vector_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Tune at session level
SET ivfflat.probes = 10;
```

## 14. Debugging Retrieval
1. Ensure rows exist: `/vector_stats`.
2. Use `/debug_retrieve?q=...&k=8`.
3. If zero hits but terms exist in corpus, inspect logs for `operator does not exist: vector <=> numeric[]` (now fixed by literal embedding).
4. Adjust `RETRIEVAL_MIN_SCORE`, `VECTOR_K`, temporarily disable grounding.
5. Add more domain documents (content coverage beats threshold tweaks).

## 15. Safety & Grounding Tuning (WIP)
| Goal | Change |
|------|--------|
| Reduce false blocks | Set `SAFETY_DISABLE_BLOCK=1` temporarily |
| Allow exploratory answers | `ENFORCE_GROUNDED=0` during tuning |
| Stricter grounding | Increase `GROUNDING_MIN_OVERLAP` (e.g., 0.12) |
| More recall | Lower `RETRIEVAL_MIN_SCORE`, raise `VECTOR_K` |

## 16. Future Enhancements
- Lexical fallback (pg_trgm) when vector hits empty.
- Hybrid scoring (BM25 + vector) reranking.
- Embedding provider abstraction (local models / Azure OpenAI / HuggingFace).
- Adaptive thresholding (percentile-based per day).
- Automatic index parameter tuning (lists, probes) using sample queries.
- Cached rephrase results with TTL to save tokens.
- Structured citation formatting / highlighting in UI.

## 17. Security / Operational Notes (FS)

## 18. Directory Structure 
```
chat_api_v1/          # Gateway
kmapi_v1/             # Retrieval + orchestration
llmapi_v1/            # LLM & embedding proxy + safety
ingestapi_v1/         # Ingestion & chunking
chatui_v1/            # Streamlit UI
postgres-init/        # Extension & (optional) index scripts
scripts/normalize_vectors.py
```

## 19. Quick Checklist for New Deployment
1. Bring up Postgres with pgvector extension (init scripts create extension).
2. Set valid `OPENAI_API_KEY` in `llmapi_v1/.env`.
3. `docker compose up -d --build`.
4. Ingest sample documents.
5. Hit `/debug_retrieve` to validate retrieval.
6. Query `/process` with trace to confirm full pipeline.
7. (Optional) Create IVF / HNSW index once corpus grows.

## 20. Troubleshooting Table (FS)
| Symptom | Action |
|---------|--------|
| 0 hits, rows > 0 | Check logs for operator error; ensure vector literal fix deployed | testing
| Slow retrieval | Add IVF index, tune lists/probes |
| Hallucinations | Raise `GROUNDING_MIN_OVERLAP`, enforce grounding |
| Over-blocking | Enable `SAFETY_DISABLE_BLOCK`, review categories |
| Low recall | Lower `RETRIEVAL_MIN_SCORE`, raise `VECTOR_K`, add docs |

## 21. License / Usage
Internal prototype.

---
