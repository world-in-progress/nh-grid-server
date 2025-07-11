from pydantic import BaseModel

class BaseChat(BaseModel):
    query: str

class BaseChatResponse(BaseModel):
    response: str