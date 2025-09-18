# Setup Guide

Concise, action‑oriented instructions for running the RAG microservices platform from scratch.

---
## 1. Prerequisites
| Requirement | Notes |
|-------------|-------|
| Docker + Docker Compose | Recommended path (simplest) |
| Python 3.11 (optional) | Only needed for running scripts locally outside containers |
| OpenAI API Key | Required for embedding + chat via `llmapi_v1` (set in `.env`) |

> Local (non-Docker) mode is possible but you must replicate per‑service env variables and install requirements manually; Docker flow below is assumed.

---
## 2. Clone Repository
```powershell
git clone <your-fork-or-repo-url>
cd chatbot_2
```

---
## 3. Environment Files
Each service has its own `.env`. Minimal required values (example):

### `llmapi_v1/.env`
```
OPENAI_API_KEY=sk-...
PORT=8002
```

### `kmapi_v1/.env`
```
LLM_API_URL=http://llmapi_v1:8002
DATABASE_URL=postgresql+psycopg2://postgres:123@postgres_v1:5432/chatdb
VECTOR_K=8
RETRIEVAL_MIN_SCORE=0.15
ENFORCE_GROUNDED=0
TRACE_DEFAULT=1
INCLUDE_DEBUG_ANSWER=1
NORMALIZE_EMBEDDINGS=1
```

### `ingestapi_v1/.env`
```
DATABASE_URL=postgresql+psycopg2://postgres:123@postgres_v1:5432/chatdb
OPENAI_API_KEY=sk-...
NORMALIZE_EMBEDDINGS=1
```

### `chat_api_v1/.env`
(If exists; often minimal)
```
PORT=8000
```

### `chatui_v1/.env`
```
API_BASE=http://chat_api_v1:8000
```

> Never commit real API keys. Use placeholders when sharing the repo.

---
## 4. Start the Stack
```powershell
docker compose up -d --build
```
Wait until containers healthy. Check:
```powershell
docker compose ps
```

---
## 5. Smoke Test
```powershell
curl http://localhost:8082/health      # llmapi_v1
curl http://localhost:8081/health      # kmapi_v1
curl http://localhost:8083/health      # ingestapi_v1
curl http://localhost:8080/health      # chat_api_v1 (if implemented)
```
Open UI: http://localhost:8502

---
## 6. Ingest Documents
### Option A: Upload PDF / DOCX
Use a tool like curl or Postman:
```powershell
curl -F "file=@sample.pdf" -F "thread_id=thread1" http://localhost:8083/upload
```

### Option B: JSON Direct Ingest
```powershell
$body = '{"thread_id":"thread1","chunk_size":700,"chunk_overlap":100,"docs":[{"text":"Billing help text ...","source":"billing.txt"}]}'
curl -X POST http://localhost:8083/ingest -H "Content-Type: application/json" -d $body
```
Verify row count:
```powershell
curl http://localhost:8083/vector_stats
```

---
## 7. Query the System
Minimal RAG:
```powershell
curl -X POST http://localhost:8081/rag_simple -H "Content-Type: application/json" -d '{"user_input":"how to pay bill","trace":true}'
```
Full pipeline:
```powershell
curl -X POST http://localhost:8081/process -H "Content-Type: application/json" -d '{"user_input":"payment receipt not received","trace":true}'
```
Debug retrieval:
```powershell
curl "http://localhost:8081/debug_retrieve?q=payment%20receipt&k=8"
```

---
## 8. Common Environment Tunables
| Variable | Effect | Suggested Dev |
|----------|--------|---------------|
| VECTOR_K | Number of chunks retrieved | 6–8 |
| RETRIEVAL_MIN_SCORE | Low-confidence gate | 0.15 (lower for recall) |
| ENFORCE_GROUNDED | Apply grounding heuristic | 0 during tuning, 1 in prod |
| GROUNDING_MIN_OVERLAP | Overlap threshold | 0.05–0.10 |
| TRACE_DEFAULT | Always include trace events | 1 for debugging |
| INCLUDE_DEBUG_ANSWER | Show raw answer (pre moderation) | 1 dev, 0 prod |
| NORMALIZE_EMBEDDINGS | Unit-length embeddings | 1 |

---
## 9. Normalizing Existing Data (Optional)
If you ingested before enabling normalization:
```powershell
$env:DATABASE_URL="postgresql://postgres:123@localhost:5543/chatdb"
python scripts/normalize_vectors.py --dry-run
python scripts/normalize_vectors.py
```

---
## 10. Indexing for Scale (Optional)
```sql
CREATE INDEX IF NOT EXISTS vector_chunks_embedding_ivfflat_cos
  ON vector_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Session tuning
SET ivfflat.probes = 10;
```

---
## 11. Troubleshooting Quick Table
| Symptom | Check | Fix |
|---------|-------|-----|
| 0 hits on debug_retrieve | `/vector_stats` row_count | Ingest docs with those terms |
| operator does not exist: vector <=> numeric[] | Retrieval logs | Ensure literal vector SQL version deployed |
| All answers "I don't know" | Threshold / grounding | Lower `RETRIEVAL_MIN_SCORE`, disable grounding temp |
| Slow retrieval | No index | Add IVF/HNSW index |

---
## 12. Clean Shutdown & Reset
```powershell
docker compose down        # stop
# Remove volumes (DESTRUCTIVE)
docker compose down -v
```

---
## 13. Local (Non-Docker) Dev (Optional)
Example for `kmapi_v1`:
```powershell
cd kmapi_v1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
uvicorn main:app --reload --port 8001
```
Ensure dependent services running (llmapi, Postgres). Adjust `LLM_API_URL` + `DATABASE_URL` in `.env`.

---
## 14. Updating Prompts
Edit JSON under `prompts/` or service `prompts/` subfolder. Hot reload requires container rebuild unless mounted.

---
