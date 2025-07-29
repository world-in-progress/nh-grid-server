from pydantic import BaseModel

class CreateCommonBody(BaseModel):
    name: str
    type: str
    src_path: str

class CopyToBody(BaseModel):
    target_path: str