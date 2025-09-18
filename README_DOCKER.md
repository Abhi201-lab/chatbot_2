This repository contains multiple microservices. Use Docker Compose to build and run them together.

Services and default container ports (host bindings in compose file may differ):
- chat_api_v1 (FastAPI gateway / chat orchestration): 8000
- kmapi_v1 (Knowledge Manager RAG + safety + grounding): 8001
- llmapi_v1 (LLM utilities: embed, chat, synthesize, safety inspect): 8002
- ingestapi_v1 (Document ingestion & chunk embedding into Postgres/pgvector): 8003
- chatui_v1 (Streamlit UI): 8501

Quick start (Windows PowerShell):

# Build and start all services
docker-compose up --build -d

# View logs for a service
docker-compose logs -f chat_api

Notes:
- Vector backend is now ONLY pgvector (Postgres). No FAISS index is used or mounted.
- Embeddings are stored in table `vector_chunks(embedding vector(1536))`.
- Each service expects an `.env` file in its folder (see compose `env_file` entries).
- To stop and remove containers:

docker-compose down

### Vector / Retrieval Details

- Query flow (full /process): safety_pre -> intent -> rephrase -> embed (LLM API /embed) -> pgvector similarity search -> answer synthesis -> post_safety -> grounding.
-Simple flow`/rag_simple` performs: embed -> pgvector retrieval -> answer .

### Embedding Normalization (Cosine Similarity)

We normalize embeddings at ingest time (L2 -> unit vectors) when `NORMALIZE_EMBEDDINGS=1`.

If you have legacy (non‑normalized) data, choose ONE path:

Option A – Re-ingest (simplest)
```sql
TRUNCATE vector_chunks;  -- then re-upload documents
```

Option B – In-place normalization script
Use the provided script (no SQL math on vectors, since pgvector doesn't support `/` arithmetic):
```powershell
$env:DATABASE_URL="postgresql://postgres:123@localhost:5543/chatdb"
python scripts/normalize_vectors.py --dry-run   # inspect how many would update
python scripts/normalize_vectors.py             # perform updates
```
Script batches updates and skips already unit-length vectors.

After normalization (or fresh ingest), create a cosine index (optional but recommended for scale):
```sql
CREATE INDEX IF NOT EXISTS vector_chunks_embedding_ivfflat_cos
  ON vector_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Session tuning
SET ivfflat.probes = 10;  -- raise for better recall (e.g. 50-200)
```

If pgvector >= 0.5.0 and you prefer HNSW:
```sql
CREATE INDEX IF NOT EXISTS vector_chunks_embedding_hnsw_cos
  ON vector_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
SET hnsw.ef_search = 64;
```

### Environment Variables (Key)

Required:
- `DATABASE_URL` (Postgres with pgvector extension)
- `LLM_API_URL` (Base URL for llmapi_v1)

Optional:
- `VECTOR_K` (top-K retrieval, default 4)
- `TRACE_DEFAULT` (set to 1 to enable tracing for all requests unless overridden)
- `GROUNDING_MIN_OVERLAP`, `RETRIEVAL_MIN_SCORE` (heuristics tuning)
- `INCLUDE_DEBUG_ANSWER` (include raw pre-moderation answer when sanitized)
- `ENFORCE_GROUNDED` (disable to allow ungrounded answers)
- `SAFETY_DISABLE_BLOCK` (bypass blocking but still returns categories)

Removed / Legacy (no longer used):
- `VECTOR_BACKEND`, `VECTOR_DB_PATH` (FAISS removed)

### Operational Tips

- Use `/vector_stats` on kmapi_v1 and ingestapi_v1 to see row counts.
- Regularly `ANALYZE vector_chunks;` after large ingests.
- For large corpora, increase IVF `lists` ~ sqrt(N) and adjust `ivfflat.probes`.

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Empty retrieval results | No rows or very low similarity | Check ingestion logs; verify embeddings inserted |
| Slow queries | Missing index or low lists/probes | Create IVF/HNSW index; tune params |
| High DB CPU | Excessive probes or large K | Reduce probes/K; add index; scale vertically |
| Ungrounded answers | Grounding disabled or threshold too low | Ensure `ENFORCE_GROUNDED=1`; tune `GROUNDING_MIN_OVERLAP` |

---
This documentation reflects the pgvector-only architecture as of current iteration.
