from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Text, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    thread_id  = Column(UUID(as_uuid=True), primary_key=True)
    message_id = Column(UUID(as_uuid=True), primary_key=True)
    user_input = Column(Text, nullable=False)
    bot_output = Column(Text, nullable=True)
    timestamp  = Column(TIMESTAMP, server_default=func.now())

class Feedback(Base):
    __tablename__ = "feedback"
    feedback_id  = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id    = Column(UUID(as_uuid=True), nullable=False)
    message_id   = Column(UUID(as_uuid=True), nullable=False)
    feedback_type = Column(Text, nullable=True)
    feedback_text = Column(Text, nullable=True)  # legacy combined text
    feedback_reasons = Column(Text, nullable=True)  # comma-separated list
    feedback_comment = Column(Text, nullable=True)
    timestamp    = Column(TIMESTAMP, server_default=func.now())
