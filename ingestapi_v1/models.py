from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class UploadMeta(BaseModel):
    thread_id: str
    uploaded_by: str | None = None

class IngestDoc(BaseModel):
    text: str = Field(..., description="Raw document text to chunk and embed")
    source: Optional[str] = Field(None, description="Source filename or identifier")
    thread_id: Optional[str] = Field(None, description="Optional thread id overriding batch thread_id")

class IngestRequest(BaseModel):
    thread_id: str = Field(..., description="Thread/workspace identifier")
    uploaded_by: Optional[str] = Field(None, description="User performing ingestion")
    chunk_size: int = Field(700, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=1000)
    docs: List[IngestDoc]

class IngestResult(BaseModel):
    status: str
    total_input_docs: int
    total_chunks: int
    vector_count_after: int
    samples: List[Dict[str, Any]]
