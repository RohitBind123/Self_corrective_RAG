from pydantic import BaseModel

class Question(BaseModel):
    input: str
