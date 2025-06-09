import json
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ....core.config import settings
from ....schemas import base, project
from ....core.bootstrapping_treeger import BT

# APIs for grid project ################################################

router = APIRouter(prefix='/project', tags=['grid / project'])

@router.get('/{name}', response_model=project.ResponseWithProjectMeta)
def get_project_meta(name: str):
    """
    Description
    --
    Retrieve project meta information by name.
    """
    
    # Check if the project file exists
    project_path = Path(settings.GRID_PROJECT_DIR, f'{name}')
    if not project_path.exists():
        raise HTTPException(status_code=404, detail='Grid project not found')
    
    # Read the project meta information from the file
    project_meta_file = project_path / settings.GRID_PROJECT_META_FILE_NAME
    try:
        with open(project_meta_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read meta information of grid project: {str(e)}')
    
    return project.ResponseWithProjectMeta(
        project_meta=project.ProjectMeta(**data)
    )

@router.post('/', response_model=base.BaseResponse)
def create_project(data: project.ProjectMeta):
    """
    Description
    --
    Create a project.
    """
    
    # Find if project already exists
    project_dir = Path(settings.GRID_PROJECT_DIR, f'{data.name}')
    if project_dir.exists():
        return base.BaseResponse(
            success=False,
            message='Grid project already exists. Please use a different name.'
        )
    
    # Create the project directory if it doesn't exist
    project_dir.mkdir(parents=True, exist_ok=True)
        
    # Write the project meta information to a file
    project_meta_path = project_dir / settings.GRID_PROJECT_META_FILE_NAME
    try:
        with open(project_meta_path, 'w') as f:
            f.write(data.model_dump_json(indent=4))
        BT.instance.mount_node('project', f'root/projects/{data.name}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save grid project meta information: {str(e)}')
    
    return base.BaseResponse(
        success=True,
        message='Grid project meta info registered successfully'
    )   
    
@router.put('/{name}', response_model=base.BaseResponse)
def update_project_info(name: str, data: project.ProjectMeta):
    """
    Description
    --
    Update meta information of a specific grid project by name.
    """
    
    # Check if the project file exists
    project_path = Path(settings.GRID_PROJECT_DIR, f'{name}')
    if not project_path.exists():
        raise HTTPException(status_code=404, detail='Grid project not found')
    
    # Write the updated project meta info to a file
    project_meta_path = project_path / settings.GRID_PROJECT_META_FILE_NAME
    try:
        with open(project_meta_path, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update meta information of grid project: {str(e)}')
    
    return base.BaseResponse(
        success=True,
        message='Grid project meta info updated successfully'
    )

@router.delete('/{name}', response_model=base.BaseResponse)
def delete_project(name: str):
    """
    Description
    --
    Delete a grid project by name.
    """
    
    # Check if the project directory exists
    project_path = Path(settings.GRID_PROJECT_DIR, f'{name}')
    if not project_path.exists():
        raise HTTPException(status_code=404, detail='Grid project not found')
    
    # TODO: Is it possible to check if the project is running?
    
    # Delete the folder of this project
    try:
        shutil.rmtree(project_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete grid project: {str(e)}')
    
    return base.BaseResponse(
        success=True,
        message='Grid project deleted successfully'
    )
        