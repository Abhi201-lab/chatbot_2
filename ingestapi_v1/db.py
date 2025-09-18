from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from environment import env
from orm import Base

DATABASE_URL = env("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db():
    """Initialize database objects.

    Adds safety for pgvector deployment in existing databases by:
      1. Creating the pgvector extension if available (idempotent)
      2. Ensuring the vector_chunks table is created
      3. Adding embedding column if the table predates extension (idempotent)
    """
    try:
        if engine.dialect.name == 'postgresql':
            with engine.connect() as conn:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                except Exception:
                    # Non-fatal: extension may require superuser or already present
                    pass
    except Exception:
        pass

    # Create tables (will skip existing)
    Base.metadata.create_all(bind=engine)

    # Backfill embedding column if table existed before extension
    try:
        if engine.dialect.name == 'postgresql':
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS embedding vector(1536)"))
    except Exception:
        # Ignore; any real issues will surface during insert
        pass
