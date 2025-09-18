from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    """Chat request; thread_id and message_id may be omitted to let backend generate them.

    Backward compatibility: existing clients still sending both IDs continue to work.
    """
    thread_id: Optional[str] = Field(None, min_length=8)
    message_id: Optional[str] = Field(None, min_length=8)
    user_input: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    thread_id: str
    message_id: str
    bot_output: str
    citations: list[dict] = []

class FeedbackRequest(BaseModel):
    thread_id: str
    message_id: str
    feedback_type: str | None = None  # 'up' or 'down'
    feedback_reasons: list[str] | None = None  # optional structured reasons
    feedback_comment: str | None = None  # optional free-form comment

    def combined_text(self) -> str | None:
        parts: list[str] = []
        if self.feedback_reasons:
            parts.append("reasons=" + ",".join(self.feedback_reasons))
        if self.feedback_comment:
            parts.append("comment=" + self.feedback_comment)
        if not parts:
            return None
        return " | ".join(parts)
