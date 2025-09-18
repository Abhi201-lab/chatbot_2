import logging
from typing import List, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

log = logging.getLogger(__name__)


class PGVectorRetriever:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
       

    def similarity_search(self, query_emb: list[float], k: int = 4) -> List[Tuple[str, float, str, str]]:
        """Return list of (content, score, source, id). Uses cosine distance."""
        session = self.SessionLocal()
        try:
            log.info("similarity_search start dim=%d k=%d", len(query_emb), k)
            vec_literal = "[" + ",".join(f"{float(v):.8f}" for v in query_emb) + "]"
            sql = text(f"""
                SELECT id, source, content, 1 - (embedding <=> '{vec_literal}') AS similarity
                FROM vector_chunks
                ORDER BY embedding <=> '{vec_literal}'
                LIMIT :k
            """)
            rows = session.execute(sql, {"k": k}).fetchall()
            out = []
            for r in rows:
                out.append((r.content, float(r.similarity), r.source, r.id))
            return out
        except Exception:
            log.exception("pgvector similarity search failed")
            return []
        finally:
            session.close()
