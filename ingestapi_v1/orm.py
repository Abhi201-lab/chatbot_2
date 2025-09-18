from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Text, TIMESTAMP, func, Integer
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    file_id    = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id  = Column(Text, nullable=False)
    filename   = Column(Text, nullable=False)
    file_type  = Column(Text, nullable=True)
    status     = Column(Text, nullable=False, default="pending")
    uploaded_by= Column(Text, nullable=True)
    timestamp  = Column(TIMESTAMP, server_default=func.now())


class VectorChunk(Base):
    __tablename__ = "vector_chunks"
    id          = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id   = Column(Text, nullable=True)
    source      = Column(Text, nullable=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    content     = Column(Text, nullable=False)
    embedding   = Column(Vector(1536), nullable=False)  # match OpenAI ada-002 dims
    created_at  = Column(TIMESTAMP, server_default=func.now())
