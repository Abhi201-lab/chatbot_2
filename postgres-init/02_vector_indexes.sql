-- Optional pgvector index creation (cosine similarity).
-- Execute automatically on first container init, or run manually later.
-- Adjust `lists` based on data size (start ~sqrt(N)).

-- IVF Flat cosine index
CREATE INDEX IF NOT EXISTS vector_chunks_embedding_ivfflat_cos
  ON vector_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- For better recall (trade memory & build time) increase lists and set probes at session level:
--   SET ivfflat.probes = 10;       -- raise to 50-200 for higher recall

-- If pgvector >= 0.5.0, you can alternatively use HNSW (uncomment if desired):
-- CREATE INDEX IF NOT EXISTS vector_chunks_embedding_hnsw_cos
--   ON vector_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
-- At query time:
--   SET hnsw.ef_search = 64;
