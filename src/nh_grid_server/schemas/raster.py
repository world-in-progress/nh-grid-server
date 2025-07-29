from enum import Enum
from typing import Any
from pydantic import BaseModel

from icrms.iraster import RasterOperation

class CreateRasterBody(BaseModel):
    name: str
    type: str
    original_tif_path: str

class GetCogTifResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] = None

class UpdateByFeatureItem(BaseModel):
    feature_node_key: str
    operation: RasterOperation
    value: float = 0.0

class UpdateByFeatureBody(BaseModel):
    updates: list[UpdateByFeatureItem]

class SamplingResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] = None

class GetMetadataResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] = None