import numpy as np
from pydantic import BaseModel

class UploadBody(BaseModel):
    file_path: str
    file_type: str
    
    