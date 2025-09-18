from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uuid, os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from environment import load_env, env
from logger import get_logger
from db import SessionLocal, init_db
from orm import Document, VectorChunk
from langchain_openai import OpenAIEmbeddings
from models import IngestRequest, IngestResult

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_env()
log = get_logger("ingestion_api")

DATABASE_URL = env("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# FAISS removed: using pgvector only
OPENAI_EMB_MODEL = env("EMBED_MODEL", default="text-embedding-ada-002", required=False)
_emb_client = OpenAIEmbeddings(model=OPENAI_EMB_MODEL)
NORMALIZE_EMBEDDINGS = env("NORMALIZE_EMBEDDINGS", default="1", required=False) in ("1","true","TRUE","yes")

app = FastAPI(title="Data Ingestion API", version="1.0.0")
init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info(f"Incoming request: {request.method} {request.url}")
    try:
        resp = await call_next(request)
        log.info(f"Request completed: {request.method} {request.url} status={resp.status_code}")
        return resp
    except Exception:
        log.exception("Unhandled exception in request pipeline")
        return {"detail": "Internal Server Error"}


@app.post("/upload")
async def upload_doc(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    uploaded_by: str = Form(None),
):
    db: Session = next(get_db())
    file_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{file_id}_{file.filename}"

    # log initially
    log.info(f"Upload started: {file.filename} by {uploaded_by} thread={thread_id}")
    doc = Document(file_id=file_id, thread_id=thread_id, filename=file.filename,
                   file_type=file.content_type, uploaded_by=uploaded_by, status="pending")
    db.add(doc)
    db.commit()

    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            log.warning("Unsupported file type: %s", file.filename)
            db.query(Document).filter_by(file_id=file_id).update({"status": "failed"})
            db.commit()
            return JSONResponse(status_code=400, content={"detail": "Unsupported file type (.pdf or .docx only)."})

        docs = loader.load()
        if not docs:
            log.warning("No text extracted from document: %s", file.filename)
            db.query(Document).filter_by(file_id=file_id).update({"status": "failed"})
            db.commit()
            return JSONResponse(status_code=400, content={"detail": "No text extracted from document."})

        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        texts = [c.page_content for c in chunks]
        metas = []
        for c in chunks:
            md = c.metadata.copy() if c.metadata else {}
            md.update({"file_id": file_id, "thread_id": thread_id, "source": file.filename})
            metas.append(md)

        # Persist to pgvector table 
        try:
            embeddings = _emb_client.embed_documents(texts)
            if NORMALIZE_EMBEDDINGS:
                import math
                normed = []
                for emb in embeddings:
                    s = sum(e*e for e in emb) or 1.0
                    l2 = math.sqrt(s)
                    normed.append([e / l2 for e in emb])
                embeddings = normed
            vc_rows = []
            for idx, (txt, meta, emb) in enumerate(zip(texts, metas, embeddings)):
                vc_rows.append(VectorChunk(
                    thread_id=meta.get("thread_id"),
                    source=meta.get("source"),
                    chunk_index=idx,
                    content=txt,
                    embedding=emb
                ))
            db.bulk_save_objects(vc_rows)
            db.commit()
        except Exception:
            log.exception("Failed to persist vector chunks to pgvector table (continuing)")

        db.query(Document).filter_by(file_id=file_id).update({"status": "ingested"})
        db.commit()
        log.info("Ingestion complete: %s chunks=%d", file.filename, len(chunks))
        return {"status": "success", "file_id": file_id, "chunks": len(chunks)}

    except Exception:
        log.exception("Ingestion failed")
        try:
            db.query(Document).filter_by(file_id=file_id).update({"status": "failed"})
            db.commit()
        except Exception:
            log.warning("Failed updating document status to failed")
        return JSONResponse(status_code=502, content={"detail": "Ingestion failed. See logs."})
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            log.warning("Temp cleanup failed")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/vector_stats")
def vector_stats():
    from sqlalchemy import text
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM vector_chunks")).scalar() or 0
    return {"row_count": int(count)}


@app.post("/cleanup_placeholders")
def cleanup_placeholders():
    return {"removed": 0, "note": "FAISS removed; no placeholders"}


@app.post("/ingest", response_model=IngestResult)
def ingest_json(req: IngestRequest):
    """Ingest raw document texts (already in memory) with chunking & embedding.

    Steps (all logged):
      1. Validate request
      2. Iterate docs, split into chunks
      3. Accumulate texts + metadata
      4. Embed & store via FAISSWriter.merge
      5. Return counts & samples
    """
    log.info("INGEST START: docs=%d chunk_size=%d overlap=%d thread=%s", len(req.docs), req.chunk_size, req.chunk_overlap, req.thread_id)
    if not req.docs:
        raise HTTPException(status_code=400, detail="No documents provided")

    splitter = RecursiveCharacterTextSplitter(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    total_chunks = 0
    texts, metas = [], []

    for idx, d in enumerate(req.docs):
        base_thread = d.thread_id or req.thread_id
        src = d.source or f"doc_{idx}"
        text = d.text.strip()
        if not text:
            log.warning("Doc %d empty text skipped", idx)
            continue
        log.info("DOC %d: source=%s len=%d", idx, src, len(text))
        # Create a temporary pseudo document structure for splitter
        from langchain.schema import Document as LCDocument
        lc_doc = LCDocument(page_content=text, metadata={"source": src, "thread_id": base_thread})
        chunks = splitter.split_documents([lc_doc])
        log.info("DOC %d: chunked into %d chunks", idx, len(chunks))
        for c_i, c in enumerate(chunks):
            texts.append(c.page_content)
            md = (c.metadata or {}).copy()
            md.update({"chunk_index": c_i, "original_source": src})
            metas.append(md)
        total_chunks += len(chunks)

    if not texts:
        raise HTTPException(status_code=400, detail="No non-empty documents to ingest")

    log.info("EMBED START: total_chunks=%d", total_chunks)
    try:
        embeddings = _emb_client.embed_documents(texts)
        if NORMALIZE_EMBEDDINGS:
            import math
            normed = []
            for emb in embeddings:
                s = sum(e*e for e in emb) or 1.0
                l2 = math.sqrt(s)
                normed.append([e / l2 for e in emb])
            embeddings = normed
        db: Session = next(get_db())
        vc_rows = []
        for idx,(txt, meta, emb) in enumerate(zip(texts, metas, embeddings)):
            vc_rows.append(VectorChunk(
                thread_id=meta.get("thread_id"),
                source=meta.get("original_source"),
                chunk_index=meta.get("chunk_index", idx),
                content=txt,
                embedding=emb
            ))
        db.bulk_save_objects(vc_rows)
        db.commit()
    except Exception:
        log.exception("Embedding/storage failed")
        raise HTTPException(status_code=502, detail="Embedding/storage failed")
    log.info("EMBED COMPLETE: inserted_rows=%d", len(texts))

    samples = []
    return IngestResult(
        status="success",
        total_input_docs=len(req.docs),
        total_chunks=total_chunks,
        vector_count_after=len(texts),
        samples=samples,
    )
