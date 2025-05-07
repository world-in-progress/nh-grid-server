import json
from pydantic import BaseModel, field_validator

class ProjectSchema(BaseModel):
    """Schema for project init configuration"""
    name: str # name of the grid schema
    epsg: int # EPSG code for the grid
    starred: bool # whether the grid schema is starred
    description: str # description of the grid schema
    base_point: tuple[float, float] # [lon, lat], base point of the grid
    grid_info: list[tuple[float, float]] # [(width_in_meter, height_in_meter), ...], grid size in each level

    @field_validator('base_point')
    def validate_base_point(cls, v):
        if len(v) != 2:
            raise ValueError('base_point must have exactly 2 values [lon, lat]')
        return v
    
    @field_validator('grid_info')
    def validate_grid_info(cls, v):
        if not all(len(item) == 2 for item in v):
            raise ValueError('grid_info must contain tuples of exactly 2 values [width_in_meter, height_in_meter]')
        return v

    @staticmethod
    def parse_file(file_path: str) -> 'ProjectSchema':
        """Parse a grid schema from a JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return ProjectSchema(**data)
    
class ResponseWithProjectSchema(BaseModel):
    """Response schema for project operations with project schema"""
    project_schema: ProjectSchema | None

    @field_validator('project_schema')
    def validate_schema(cls, v):
        if v is None:
            return v
        # Ensure that the schema is an instance of ProjectSchema
        if not isinstance(v, ProjectSchema):
            raise ValueError('schema must be an instance of ProjectSchema')
        return v

class ResponseWithProjectSchemas(BaseModel):
    """Response schema for project operations with project schemas"""
    project_schemas: list[ProjectSchema] | None

    @field_validator('project_schemas')
    def validate_schemas(cls, v):
        if v is None:
            return v
        # Ensure that the schemas are instances of ProjectSchema
        if not all(isinstance(schema, ProjectSchema) for schema in v):
            raise ValueError('schemas must be a list of ProjectSchema instances')
        return v
 