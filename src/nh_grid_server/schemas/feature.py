import numpy as np
from pydantic import BaseModel
from typing import Any

class UploadBody(BaseModel):
    file_path: str
    file_type: str
    feature_type: str

class FeatureSaveBody(BaseModel):
    feature_name: str
    feature_type: str
    feature_json: dict[str, Any]

class UploadedFeatureSaveBody(BaseModel):
    file_path: str
    feature_type: str
    feature_json: dict[str, Any]
    is_edited: bool
    
class GetFeatureJsonInfo(BaseModel):
    feature_name: str
    feature_type: str
    