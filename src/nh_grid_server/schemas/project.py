from pathlib import Path
from pydantic import BaseModel, field_validator
from ..core.config import settings

class SubprojectMeta(BaseModel):
    """Information about the subproject of a specific project"""
    name: str
    starred: bool # whether the subproject is starred
    description: str # description of the subproject
    bounds: tuple[float, float, float, float] # [ min_lon, min_lat, max_lon, max_lat ] 
    
    @field_validator('bounds')
    def validate_bounds(cls, v):
        if len(v) != 4:
            raise ValueError('bounds must have exactly 4 values [min_lon, min_lat, max_lon, max_lat]')
        return v

class ProjectMeta(BaseModel):
    """Information about the project"""
    name: str
    starred: bool # whether the project is starred
    description: str # description of the project
    schema_name: str # name of project schema the project is based on

class ResponseWithProjectMeta(BaseModel):
    """Response schema for project meta info"""
    project_meta: ProjectMeta | None

    @field_validator('project_meta')
    def validate_project_meta(cls, v):
        if v is None:
            return v
        # Ensure that the project_meta is an instance of ProjectMeta
        if not isinstance(v, ProjectMeta):
            raise ValueError('project_meta must be an instance of ProjectMeta')
        return v
    
class ResponseWithSubprojectMetas(BaseModel):
    """Response meta information for subprojects"""
    subproject_metas: list[SubprojectMeta] | None

    @field_validator('subproject_metas')
    def validate_subproject_metas(cls, v):
        if v is None:
            return v
        # Ensure that the subproject_metas are instances of SubprojectMeta
        if not all(isinstance(info, SubprojectMeta) for info in v):
            raise ValueError('subproject_metas must be a list of SubprojectMeta instances')
        return v

class ResponseWithProjectMetas(BaseModel):
    """Response schema for project meta info"""
    project_metas: list[ProjectMeta] | None

    @field_validator('project_metas')
    def validate_project_metas(cls, v):
        if v is None:
            return v
        # Ensure that the project_metas are instances of ProjectMeta
        if not all(isinstance(info, ProjectMeta) for info in v):
            raise ValueError('project_metas must be a list of ProjectMeta instances')
        return v

class ProjectStatus(BaseModel):
    """Status of the project"""
    status: str # 'ACTIVATED', 'DEACTIVATED'
    is_ready: bool # True if the project is ready to be used

    @field_validator('status')
    def validate_status(cls, v):
        if v not in ['ACTIVATED', 'DEACTIVATED']:
            raise ValueError('status must be either "ACTIVATED" or "DEACTIVATED"')
        return v
    
    @staticmethod
    def activated_status():
        return 'ACTIVATED'
    
    @staticmethod
    def deactivated_status():
        return 'DEACTIVATED'
