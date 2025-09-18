from pydantic import BaseModel
from typing import Optional

class KMRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None
    message_id: Optional[str] = None
    trace: Optional[bool] = False
