from typing import Any
from pydantic import BaseModel

from icrms.iraster import RasterOperation

class CreateRasterBody(BaseModel):
    name: str
    original_tif_path: str

class GetCogTifResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] = None

class UpdateByFeatureBody(BaseModel):
    feature: dict[str, Any]
    operation: RasterOperation
    value: float = 0.0

class SamplingResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] = None

class GetMetadataResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] = None