import json
import c_two as cc
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ....core.config import settings
from ....schemas.base import BaseResponse
from ....core.server import set_current_project
from ....schemas.project import ProjectMeta, ProjectStatus, SubprojectMeta

# APIs for grid subproject ################################################

router = APIRouter(prefix='/subproject', tags=['grid / subproject'])

@router.get('/', response_model=ProjectStatus)
def check_subproject_ready():
    """
    Description
    --
    Check if the subproject runtime resource is ready.
    """
    
    try:
        flag = cc.message.Client.ping(settings.TCP_ADDRESS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to check CRM of the subproject: {str(e)}')
    
    return ProjectStatus(
        status='ACTIVATED' if flag else 'DEACTIVATED',
        is_ready=flag
    )

@router.get('/{project_name}/{subproject_name}', response_model=BaseResponse)
def set_subproject(project_name: str, subproject_name: str):
    """
    Description
    --
    Set a specific subproject as the current crm server.
    """
    
    # Check if the subproject directory exists
    project_dir = Path(settings.PROJECT_DIR, project_name)
    subproject_dir = project_dir / subproject_name
    if not subproject_dir.exists():
        raise HTTPException(status_code=404, detail=f'Grid subproject ({subproject_name}) belonging to project ({project_name}) not found')

    try:
        project_meta_file = project_dir / settings.GRID_PROJECT_META_FILE_NAME
        with open(project_meta_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')
    
    project_meta = ProjectMeta(**data)
    try:
        set_current_project(project_meta, subproject_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set subproject as the current resource: {str(e)}')
    return BaseResponse(
        success=True,
        message='Grid subproject set successfully'
    )

@router.post('/{project_name}', response_model=BaseResponse)
def create_subproject(project_name: str, data: SubprojectMeta):
    """
    Description
    --
    Create a subproject belonging to a specified project.
    """
    
    # Check if the project directory exists
    project_dir = Path(settings.PROJECT_DIR, project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f'Grid project ({project_name}) not found')

    # Check if the subproject directory already exists
    subproject_dir = project_dir / data.name
    if subproject_dir.exists():
        return BaseResponse(
            success=False,
            message='Grid subproject already exists. Please use a different name.'
        )
    
    # Write the subproject meta information to a file
    subproject_dir.mkdir(parents=True, exist_ok=True)
    subproject_meta_file = subproject_dir / settings.GRID_SUBPROJECT_META_FILE_NAME
    try:
        with open(subproject_meta_file, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save grid subproject meta information: {str(e)}')
    
    return BaseResponse(
        success=True,
        message='Grid subproject created successfully'
    )
    
@router.put('/{project_name}/{subproject_name}', response_model=BaseResponse)
def update_subproject(project_name: str, subproject_name: str, data: SubprojectMeta):
    """
    Description
    --
    Update a specific subproject by new meta information.
    """
    
    # Check if the subproject directory exists
    project_dir = Path(settings.PROJECT_DIR, project_name)
    subproject_dir = project_dir / subproject_name
    if not subproject_dir.exists():
        raise HTTPException(status_code=404, detail=f'Subproject ({subproject_name}) belonging to project ({project_name}) not found')
    
    # Write the updated subproject meta information to a file
    subproject_meta_file = subproject_dir / settings.GRID_SUBPROJECT_META_FILE_NAME
    try:
        with open(subproject_meta_file, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update grid subproject meta information: {str(e)}')
    
    return BaseResponse(
        success=True,
        message='Grid subproject updated successfully'
    )

@router.delete('/{project_name}/{subproject_name}', response_model=BaseResponse)
def delete_project(project_name: str, subproject_name: str):
    """
    Description
    --
    Delete a subproject by specific names of project and subproject.
    """
    
    # Check if the subproject directory exists
    project_dir = Path(settings.PROJECT_DIR, project_name, subproject_name)
    subproject_dir = project_dir / subproject_name
    if not subproject_dir.exists():
        raise HTTPException(status_code=404, detail='Subproject not found')
    
    # Delete the subproject directory
    try:
        for item in subproject_dir.iterdir():
            item.unlink()
        subproject_dir.rmdir()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete subproject ({subproject_name}) belonging to project ({project_name}): {str(e)}')
    
    return BaseResponse(
        success=True,
        message='Subproject deleted successfully'
    )
