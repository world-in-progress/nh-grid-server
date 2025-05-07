import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ....core.config import settings
from ....schemas.base import BaseResponse
from ....schemas.project import ProjectMeta
from ....schemas.schema import ProjectSchema, ResponseWithProjectSchema

# APIs for single project schema ##################################################

router = APIRouter(prefix='/schema', tags=['grid / schema'])

@router.get('/{name}', response_model=ResponseWithProjectSchema)
def get_schema(name: str):
    """
    Description
    --
    Get a project schema by name.
    """
    
    # Check if the schema file exists
    project_schema_path = Path(settings.SCHEMA_DIR, f'{name}.json')
    if not project_schema_path.exists():
        raise HTTPException(status_code=404, detail='Project schema not found')
    
    # Read the schema from the file
    try:
        with open(project_schema_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read schema: {str(e)}')
    
    return ResponseWithProjectSchema(
        project_schema=ProjectSchema(**data)
    )

@router.post('/', response_model=BaseResponse)
def register_schema(data: ProjectSchema):
    """
    Description
    --
    Register a project schema.
    """
    
    # Find if project schema is existed
    project_schema_path = Path(settings.SCHEMA_DIR, f'{data.name}.json')
    if project_schema_path.exists():
        return BaseResponse(
            success=False,
            message='Project schema already exists. Please use a different name.'
        )
        
    # Write the schema to a file
    try:
        with open(project_schema_path, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save project schema: {str(e)}')
    return BaseResponse(
        success=True,
        message='Project schema registered successfully'
    )

@router.put('/{name}', response_model=BaseResponse)
def update_schema(name: str, data: ProjectSchema):
    """
    Description
    --
    Update a project schema by name.
    """
    
    # Check if the schema file exists
    project_schema_path = Path(settings.SCHEMA_DIR, f'{name}.json')
    if not project_schema_path.exists():
        raise HTTPException(status_code=404, detail='Project schema not found')
    
    # Write the updated schema to the file
    try:
        with open(project_schema_path, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update schema: {str(e)}')
    
    return BaseResponse(
        success=True,
        message='Project schema updated successfully'
    )

@router.delete('/{name}', response_model=BaseResponse)
def delete_schema(name: str):
    """
    Description
    --
    Delete a project schema by name.
    """
    
    # Check if the schema file exists
    project_schema_path = Path(settings.SCHEMA_DIR, f'{name}.json')
    if not project_schema_path.exists():
        raise HTTPException(status_code=404, detail='Project schema not found')
    
    # Check if no project depends on this schema
    dependency_found = False
    project_dirs = list(Path(settings.PROJECT_DIR).glob('*'))
    for project_dir in project_dirs:
        meta_file_path = Path(project_dir, 'meta.json')
        if not meta_file_path.exists():
            continue
        
        with open(meta_file_path, 'r') as f:
            data = json.load(f)
            meta = ProjectMeta(**data)
            if meta.schema_name == name:
                dependency_found = True
                break
    if dependency_found:
        raise HTTPException(status_code=400, detail='Schema is still in use by at least one project')
    
    # Delete the schema file
    try:
        project_schema_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete schema: {str(e)}')
    
    return BaseResponse(
        success=True,
        message='Project schema deleted successfully'
    )
        