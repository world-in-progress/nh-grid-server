from typing import Any
from pydantic import BaseModel

class CreateFeatureBody(BaseModel):
    name: str
    type: str
    color: str
    epsg: str

class FeatureSaveBody(BaseModel):
    node_key: str
    feature_json: dict[str, Any]

class GetFeatureResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] | None = None

class GetFeatureJsonResponse(BaseModel):
    success: bool
    message: str
    feature_json: dict[str, Any] | None = None

class UpdateFeatureBody(BaseModel):
    name: str
    type: str
    color: str
    epsg: str
    feature_json: dict[str, Any]

class UploadFeatureSaveBody(BaseModel):
    node_key: str
    file_path: str
    file_type: str
    
class GetFeatureJsonInfo(BaseModel):
    name: str