from fastapi import FastAPI, Depends, HTTPException, Request
import time
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import requests

from environment import load_env, env
from logger import get_logger
from models import ChatRequest, ChatResponse, FeedbackRequest
import uuid
from db import SessionLocal, init_db
from orm import Conversation, Feedback


load_env()
log = get_logger("chatbot_api")
KM_API = env("KM_API_URL")

app = FastAPI(title="Chatbot API", version="1.0.0")
init_db()

@app.on_event("startup")
def log_schema_state():
    from sqlalchemy import inspect
    try:
        insp = inspect(SessionLocal().bind)
        cols = [c['name'] for c in insp.get_columns('feedback')]
        log.info(f"Feedback table columns at startup: {cols}")
    except Exception:
        log.exception("Could not inspect feedback table schema on startup")


@app.get("/health")
def health():
    return {"status": "ok"}


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
        response = await call_next(request)
        log.info(f"Completed request: {request.method} {request.url} - status {response.status_code}")
        return response
    except Exception:
        log.exception("Unhandled exception in request pipeline")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    t0 = time.time()
    # Auto-generate IDs if not provided
    thread_id = req.thread_id or str(uuid.uuid4())
    message_id = req.message_id or str(uuid.uuid4())
    log.info(f"Chat request thread={thread_id} message={message_id}")
    try:
        conv = Conversation(thread_id=thread_id, message_id=message_id, user_input=req.user_input)
        db.merge(conv)
        db.commit()

        km_elapsed = None
        try:
            km_t0 = time.time()
            forward_payload = {"thread_id": thread_id, "message_id": message_id, "user_input": req.user_input}
            r = requests.post(f"{KM_API}/process", json=forward_payload, timeout=60)
            r.raise_for_status()
            km_elapsed = time.time() - km_t0
        except requests.exceptions.ConnectionError:
            log.exception("Could not connect to KM service")
            raise HTTPException(status_code=503, detail="Knowledge service unreachable. Check KM_API and ensure the service is running.")
        except requests.exceptions.Timeout:
            log.exception("KM service timeout")
            raise HTTPException(status_code=504, detail="Knowledge service timed out.")
        except requests.exceptions.HTTPError:
            log.exception("KM service returned error")
            raise HTTPException(status_code=502, detail="Knowledge service returned an error")
        except Exception:
            log.exception("Unexpected error calling KM service")
            raise HTTPException(status_code=500, detail="Unexpected error contacting knowledge service.")

        data = r.json()
        bot_output = data.get("bot_output", "")
        citations = data.get("citations", [])

        row = db.query(Conversation).filter_by(thread_id=thread_id, message_id=message_id).first()
        if row:
            row.bot_output = bot_output
            db.commit()

        total = time.time() - t0
        log.info(f"CHAT_TIMINGS total={total:.2f}s km_call={(km_elapsed or 0):.2f}s")
        return ChatResponse(thread_id=thread_id, message_id=message_id, bot_output=bot_output, citations=citations)
    except HTTPException:
        raise
    except Exception:
        log.exception("Unexpected error in /chat handler")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    original_type = req.feedback_type
    if req.feedback_type == 'up':
        # Normalize legacy 'up' to 'positive'
        req.feedback_type = 'positive'
    log.info(
        "Feedback received thread=%s message=%s type=%s (orig=%s) reasons=%s comment_len=%s",
        req.thread_id, req.message_id, req.feedback_type, original_type, req.feedback_reasons, len(req.feedback_comment or "")
    )
    try:
        # Basic validation
        if not req.thread_id or not req.message_id:
            raise HTTPException(status_code=400, detail="thread_id and message_id required")
        if req.feedback_type not in (None, 'positive', 'down'):
            raise HTTPException(status_code=400, detail="feedback_type must be 'positive' or 'down'")

        combined = req.combined_text()
        reasons_csv = ",".join(req.feedback_reasons) if req.feedback_reasons else None
        fb = Feedback(thread_id=req.thread_id, message_id=req.message_id,
                      feedback_type=req.feedback_type,
                      feedback_text=combined,  # legacy compatibility
                      feedback_reasons=reasons_csv,
                      feedback_comment=req.feedback_comment)
        db.add(fb)
        db.commit()
        # Post-commit verification query
        count = db.query(Feedback).filter_by(thread_id=req.thread_id, message_id=req.message_id).count()
        saved = db.query(Feedback).filter_by(feedback_id=fb.feedback_id).first()
        log.info("Feedback saved id=%s verify_count=%s reasons_csv=%s saved_type=%s", fb.feedback_id, count, reasons_csv, saved.feedback_type if saved else None)
        return {"status": "ok", "feedback_id": fb.feedback_id, "verify_count": count, "saved_feedback_type": saved.feedback_type if saved else None}
    except Exception:
        log.exception("Feedback insert failed")
        return JSONResponse(status_code=500, content={"detail": "Feedback save failed"})


@app.get("/feedback/{thread_id}/{message_id}")
def get_feedback(thread_id: str, message_id: str, db: Session = Depends(get_db)):
    """Return feedback rows for a specific thread/message for debugging."""
    rows = db.query(Feedback).filter_by(thread_id=thread_id, message_id=message_id).all()
    out = []
    for r in rows:
        out.append({
            "feedback_id": r.feedback_id,
            "feedback_type": r.feedback_type,
            "feedback_text": r.feedback_text,
            "feedback_reasons": r.feedback_reasons.split(',') if r.feedback_reasons else [],
            "feedback_comment": r.feedback_comment,
            "timestamp": str(r.timestamp)
        })
    return {"count": len(out), "items": out}


# Health endpoint for compose and smoke tests
@app.get("/health")
def health():
    return {"status": "ok"}
