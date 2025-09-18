import streamlit as st
import requests, uuid, time, asyncio, httpx, hashlib
from environment import load_env, env
from logger import get_logger
from datetime import datetime

load_env()
log = get_logger("ui")

CHATBOT_API = env("CHATBOT_API_URL")
INGESTION_API = env("INGESTION_API_URL")

st.set_page_config(page_title="Ask-the-Docs", page_icon="🤖", layout="wide")

# session initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None  # will be assigned after first response
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, text, thread_id, message_id, citations, feedback, t}
if "citations_visibility" not in st.session_state:
    st.session_state.citations_visibility = {}
if "active_feedback" not in st.session_state:
    st.session_state.active_feedback = None
if "awaiting_response" not in st.session_state:
    # Indicates a question was sent and we're waiting for the API response
    st.session_state.awaiting_response = False
if "submitted_question" not in st.session_state:
    # Holds the last submitted question from the form (single-shot)
    st.session_state.submitted_question = None


# sidebar upload + new chat
with st.sidebar:
    st.markdown("## 📄 Upload Documents")

    # Initialize dedupe structure
    if "uploaded_file_hashes" not in st.session_state:
        st.session_state.uploaded_file_hashes = set()

    def file_hash(buf: bytes) -> str:
        return hashlib.md5(buf).hexdigest()  # create hash for each file to check duplicates

    with st.form("upload_form", clear_on_submit=False):
        f = st.file_uploader("File (PDF/DOCX)", type=["pdf", "docx"], key="file_uploader")
        force_reupload = st.checkbox("Force re-upload", value=False, help="Ignore duplicate detection and upload again")
        upload_submit = st.form_submit_button("Upload")

    if upload_submit:
        if f is None:
            st.warning("Select a file first.")
        else:
            raw = f.getbuffer()
            h = file_hash(raw)
            if h in st.session_state.uploaded_file_hashes and not force_reupload:
                st.info("Already uploaded this file (hash dedupe). Use 'Force re-upload' to send again.")
            else:
                try:
                    provisional_thread = st.session_state.thread_id or str(uuid.uuid4())
                    files = {"file": (f.name, raw, f.type)}
                    data = {"thread_id": provisional_thread}
                    log.info("Uploading file %s (hash=%s) to ingestion service", f.name, h)
                    with st.spinner("Uploading …"):
                        r = requests.post(f"{INGESTION_API}/upload", files=files, data=data, timeout=180)
                    if r.ok:
                        st.success("Uploaded ✅")
                        st.session_state.uploaded_file_hashes.add(h)
                        if st.session_state.thread_id is None:
                            st.session_state.thread_id = provisional_thread
                    else:
                        log.error("Ingestion service returned non-OK: %s", r.text)
                        st.error(f"Upload failed ({r.status_code})")
                except Exception:
                    log.exception("Upload failed")
                    st.error("Upload error – see logs")

    if st.button("🆕 New Chat"):
        st.session_state.thread_id = None
        st.session_state.history = []
        st.session_state.last_input = ""
        st.session_state.pending_clear = True
        st.session_state.citations_visibility = {}
        st.session_state.active_feedback = None
        st.session_state.awaiting_response = False
        for k in ["last_sent_question", "question_input"]:
            if k in st.session_state:
                del st.session_state[k]
        # Keep uploaded_file_hashes to allow reusing files in new chat

    st.markdown("---")
    st.caption("Lightweight chat UI. IDs assigned by backend after first send.")


# css
CUSTOM_CSS = """
<style>
/* Container scroll area */
.chat-scroll {max-height: 72vh; overflow-y: auto; padding-right: .5rem;}
.bubble {border-radius: 12px; padding: 0.75rem 0.9rem; margin-bottom: 0.6rem; position: relative; font-size: 0.92rem; line-height: 1.3rem;}
.bubble-user {background:#f2f4f8; color:#222; border:1px solid #e0e4ea;}
.bubble-bot {background:#e7f0ff; color:#1a1d21; border:1px solid #d2e3f7;}
.meta {font-size:0.65rem; opacity:0.6; margin-bottom:0.25rem;}
.feedback-bar {margin-top:0.35rem; font-size:0.8rem;}
.feedback-bar button {margin-right: .35rem;}
.citations {background:#ffffffdd; border:1px solid #c9d4e2; padding:0.4rem 0.6rem; border-radius:8px; margin-top:0.4rem;}
.small-btn {background:none; border:none; color:#555; cursor:pointer; font-size:0.75rem; padding:2px 6px;}
.small-btn:hover {color:#000;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Ask-the-Docs 🤖")
st.caption("Ask questions about your uploaded documents.")

# Predefine async feedback sender so its available for button handlers above
async def send_feedback_async(feedback_type: str, message_id: str, reasons=None, comment: str | None = None):
    thread_id_local = st.session_state.get("thread_id")
    if not thread_id_local:
        return
    payload = {
        "thread_id": thread_id_local,
        "message_id": message_id,
        "feedback_type": feedback_type,
        "feedback_reasons": reasons or [],
        "feedback_comment": comment or ""
    }
    log.info(f"Submitting feedback payload={payload}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"{CHATBOT_API}/feedback", json=payload)
        if r.status_code >= 400:
            log.error("Feedback API non-OK status=%s body=%s", r.status_code, r.text)
        else:
            log.info("Feedback accepted status=%s body=%s", r.status_code, r.text)
        for m in st.session_state.history:
            if m.get("message_id") == message_id:
                m["feedback"] = feedback_type
    except Exception:
        log.exception("Feedback submit failed")

# render chat history
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    for idx, m in enumerate(st.session_state.history):
        role = m["role"]
        css_class = "bubble-user" if role == "user" else "bubble-bot"
        ts = datetime.fromtimestamp(m["t"]).strftime("%H:%M:%S")
        feedback = m.get("feedback")
        citations = m.get("citations") or []
        st.markdown(f"<div class='bubble {css_class}'><div class='meta'>{'You' if role=='user' else 'Bot'} · {ts}</div><div>{m['text']}</div></div>", unsafe_allow_html=True)
        if role == "bot":
            # Feedback buttons
            if feedback is None and m.get("message_id"):
                c1, c2, c3 = st.columns([1,1,8])
                if c1.button("👍", key=f"up_{m['message_id']}"):
                    # Instant positive feedback: no form
                    # Submit minimal payload asynchronously (fire and forget)
                    try:
                        log.info(f"UI: sending positive feedback for message_id={m['message_id']}")
                        asyncio.run(send_feedback_async("positive", m["message_id"], reasons=None, comment=None))
                        log.info(f"UI: positive feedback sent for message_id={m['message_id']}")
                        m["feedback"] = "positive"
                    except Exception:
                        log.exception("Instant positive feedback failed")
                    st.rerun()
                if c2.button("👎", key=f"down_{m['message_id']}"):
                    st.session_state.active_feedback = {"message_id": m["message_id"], "type": "down"}
                    st.rerun()
            else:
                if feedback is not None:
                    summary_bits = []
                    if m.get("feedback_reasons"):
                        summary_bits.append(",".join(m["feedback_reasons"]))
                    if m.get("feedback_comment"):
                        summary_bits.append(m["feedback_comment"])
                    extra = (" – ".join(summary_bits)) if summary_bits else " submitted"
                    st.caption(f"Feedback: {'👍' if feedback in ('up','positive') else '👎'}{extra}")
            # Citations toggle
            if citations and m.get("message_id"):
                mid = m["message_id"]
                visible = st.session_state.citations_visibility.get(mid, False)
                label = ("Hide sources" if visible else "Show sources") + f" ({len(citations)})"
                if st.button(label, key=f"cit_{mid}"):
                    st.session_state.citations_visibility[mid] = not visible
                    st.rerun()
                if visible:
                    for c in citations:
                        st.markdown(f"• {c}")
    st.markdown('</div>', unsafe_allow_html=True)

# user input form
st.write("")
if "input_generation" not in st.session_state:
    # Used to force a brand new widget (empty) after successful responses
    st.session_state.input_generation = 0
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

current_input_key = f"question_input_{st.session_state.input_generation}"
with st.form(key="question_form", clear_on_submit=False):
    question = st.text_input(
        "Type your question and press Enter",
        key=current_input_key,
        value=st.session_state.last_input,
        placeholder="Ask something about the docs..."
    )
    submitted = st.form_submit_button("Send")

if submitted and question.strip():
    st.session_state.submitted_question = question.strip()
    # keep the text visible; store it
    st.session_state.last_input = question.strip()
else:
    # Do not reuse old submitted_question after we process
    if st.session_state.submitted_question is None:
        st.session_state.pending_clear = False

## send_feedback_async moved up


def process_special_command(cmd: str) -> bool:
    return False

if st.session_state.submitted_question and process_special_command(st.session_state.submitted_question):
    st.session_state.submitted_question = None
    st.stop()

if st.session_state.submitted_question:
    question = st.session_state.submitted_question
    # Clear submitted_question immediately so reruns won't reprocess
    st.session_state.submitted_question = None
    # If an API call is still in flight (streamlit rerun) don't process again
    if st.session_state.get("awaiting_response"):
        st.stop()
    # Guard: avoid duplicate resend only if no active feedback interaction
    if 'last_sent_question' in st.session_state \
            and question == st.session_state.last_sent_question \
            and st.session_state.active_feedback is None:
        st.stop()
    # Append user message locally
    st.session_state.history.append({
        "role": "user",
        "text": question,
        "thread_id": st.session_state.thread_id,
        "message_id": None,
        "citations": [],
        "feedback": None,
        "t": time.time()
    })
    placeholder_index = len(st.session_state.history)
    # placeholder bot message
    st.session_state.history.append({
        "role": "bot",
        "text": "(thinking…)",
        "thread_id": st.session_state.thread_id,
        "message_id": None,
        "citations": [],
        "feedback": None,
        "t": time.time()
    })
    user_input = question
    st.session_state.last_sent_question = question
    # preserve last_input so it stays in the box during processing
    st.session_state.last_input = user_input
    # do not clear pending_clear so input remains
    st.session_state.awaiting_response = True

    try:
        with st.spinner("Consulting knowledge base…"):
            payload = {"user_input": user_input}
            if st.session_state.thread_id:  # send thread for continuity
                payload["thread_id"] = st.session_state.thread_id
            async def chat_call():
                async with httpx.AsyncClient(timeout=60) as client:
                    return await client.post(f"{CHATBOT_API}/chat", json=payload)
            response = asyncio.run(chat_call())
            response.raise_for_status()
            data = response.json()
    except requests.HTTPError as he:
        log.error("Chat API HTTP error: %s", he.response.text if he.response is not None else str(he))
        st.session_state.history[placeholder_index-1]["text"] = user_input  # ensure user msg intact
        st.session_state.history[placeholder_index-1]["t"] = st.session_state.history[placeholder_index-1]["t"]
        st.session_state.history[placeholder_index-1]["citations"] = []
        st.session_state.history[placeholder_index]["text"] = "Error: service returned an error."
        st.session_state.awaiting_response = False
    except Exception:
        log.exception("Chat call failed")
        st.session_state.history[placeholder_index]["text"] = "Error: internal failure."
        st.session_state.awaiting_response = False
    else:
        # Successful response path
        if st.session_state.thread_id is None:
            st.session_state.thread_id = data.get("thread_id")
        st.session_state.history[placeholder_index]["text"] = data.get("bot_output", "(no answer)")
        st.session_state.history[placeholder_index]["citations"] = data.get("citations", [])
        st.session_state.history[placeholder_index]["message_id"] = data.get("message_id")
        st.session_state.history[placeholder_index]["thread_id"] = st.session_state.thread_id
    st.session_state.awaiting_response = False
    # On success: clear the remembered last_input and bump generation so a fresh empty widget renders
    st.session_state.last_input = ""
    st.session_state.input_generation += 1
    # If there was an error we leave last_input as-is so the user can edit/resend.
    st.rerun()

# feedback form user interaction
if st.session_state.active_feedback:
    with st.sidebar:
        ctx = st.session_state.active_feedback
        st.markdown("### Provide Feedback")
        reasons_catalog = ["incorrect_answer","wrong_sources","incomplete_answer","hallucination","format_issue","Other"]
        # Persist selections across reruns – keys fixed
        selected = st.multiselect("Reasons", reasons_catalog, key="fb_reasons")
        other_text = None
        if "Other" in selected:
            other_text = st.text_input("Other reason", key="fb_other")
        comment = st.text_area("Comment (optional)", key="fb_comment")
        c1, c2 = st.columns(2)
        submit_clicked = c1.button("Submit", key="fb_submit")
        cancel_clicked = c2.button("Cancel", key="fb_cancel")
        if submit_clicked:
            reasons_to_send = [r for r in selected if r != "Other"]
            if other_text:
                reasons_to_send.append(f"other:{other_text}")
            asyncio.run(send_feedback_async(ctx["type"], ctx["message_id"], reasons=reasons_to_send, comment=comment))
            for m in st.session_state.history:
                if m.get("message_id") == ctx["message_id"]:
                    m["feedback"] = ctx["type"]
                    m["feedback_reasons"] = reasons_to_send
                    m["feedback_comment"] = comment
            st.session_state.active_feedback = None
            for k in ["fb_reasons","fb_other","fb_comment"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
        elif cancel_clicked:
            st.session_state.active_feedback = None
            for k in ["fb_reasons","fb_other","fb_comment"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
