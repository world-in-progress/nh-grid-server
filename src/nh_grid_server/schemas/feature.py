import numpy as np
from pydantic import BaseModel
from typing import Any

class UploadBody(BaseModel):
    file_path: str
    file_type: str

class FeatureProperty(BaseModel):
    id: str
    name: str
    type: str
    icon: str
    symbology: str

class UpdateFeaturePropertyBody(BaseModel):
    name: str
    icon: str
    symbology: str

class FeatureSaveBody(BaseModel):
    feature_property: FeatureProperty
    feature_json: dict[str, Any]

class UploadedFeatureSaveBody(BaseModel):
    file_path: str
    feature_json: dict[str, Any]
    is_edited: bool
    
class GetFeatureJsonInfo(BaseModel):
    feature_name: str
    
class FeatureMeta(BaseModel):
    feature_meta: list[FeatureProperty]