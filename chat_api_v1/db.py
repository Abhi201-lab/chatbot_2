from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from environment import env
from orm import Base

# Single primary (container) database only
DATABASE_URL = env("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Create tables in primary (and secondary if configured)."""
    Base.metadata.create_all(bind=engine)
    # rudimentary migration for new feedback columns
    with engine.connect() as conn:
        try:
            # Check existing columns
            res = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='feedback'"))
            cols = {r[0] for r in res}
            alter_stmts = []
            if 'feedback_reasons' not in cols:
                alter_stmts.append("ALTER TABLE feedback ADD COLUMN feedback_reasons TEXT NULL")
            if 'feedback_comment' not in cols:
                alter_stmts.append("ALTER TABLE feedback ADD COLUMN feedback_comment TEXT NULL")
            for stmt in alter_stmts:
                conn.execute(text(stmt))
            if alter_stmts:
                conn.commit()
        except Exception:
            # Swallow to avoid startup failure; logs may not be configured here
            pass
    # Secondary DB support removed; all writes go to primary only.
