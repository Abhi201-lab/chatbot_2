from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class EmbedRequest(BaseModel):
    text: str

class SynthRequest(BaseModel):
    query: str
    context: str

class ChatRequest(BaseModel):
    prompt: str
