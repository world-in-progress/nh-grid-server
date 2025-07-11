from pydantic import BaseModel, field_validator

class PatchMeta(BaseModel):
    """Information about the patch of a specific project"""
    name: str
    starred: bool # whether the patch is starred
    description: str # description of the patch
    bounds: tuple[float, float, float, float] # [ min_lon, min_lat, max_lon, max_lat ] 
    
    @field_validator('bounds')
    def validate_bounds(cls, v):
        if len(v) != 4:
            raise ValueError('bounds must have exactly 4 values [min_lon, min_lat, max_lon, max_lat]')
        return v