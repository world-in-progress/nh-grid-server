import json
import c_two as cc
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ....schemas.base import BaseResponse
from ....core.bootstrapping_treeger import BT
from ....core.server import set_current_feature
from ....core.config import settings, APP_CONTEXT
from ....schemas.project import ProjectMeta, PatchStatus, PatchMeta

from icrms.itreeger import ReuseAction

# APIs for grid patch ################################################

router = APIRouter(prefix='/patch', tags=['grid / patch'])

@router.get('/', response_model=PatchStatus)
def check_patch_ready():
    """
    Description
    --
    Check if the patch runtime resource is ready.
    """
    
    try:
        node_key = f'root/projects/{APP_CONTEXT["current_project"]}/{APP_CONTEXT["current_patch"]}'
        tcp_address = BT.instance.activate_node(node_key)
        flag = cc.message.Client.ping(tcp_address)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to check CRM of the patch: {str(e)}')

    return PatchStatus(
        status='ACTIVATED' if flag else 'DEACTIVATED',
        is_ready=flag
    )

@router.get('/{project_name}/{patch_name}', response_model=BaseResponse)
def set_patch(project_name: str, patch_name: str):
    """
    Description
    --
    Set a specific patch as the current crm server.
    """
    
    # Check if the patch directory exists
    project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
    patch_dir = project_dir / patch_name
    if not patch_dir.exists():
        raise HTTPException(status_code=404, detail=f'Grid patch ({patch_name}) belonging to project ({project_name}) not found')
    
    try:
        node_key = f'root/projects/{project_name}/{patch_name}'
        APP_CONTEXT['current_project'] = project_name
        APP_CONTEXT['current_patch'] = patch_name
        BT.instance.activate_node(node_key)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set patch as the current resource: {str(e)}')
    return BaseResponse(
        success=True,
        message='Grid patch set successfully'
    )

@router.post('/{project_name}', response_model=BaseResponse)
def create_patch(project_name: str, data: PatchMeta):
    """
    Description
    --
    Create a patch belonging to a specified project.
    """

    # Check if the project directory exists
    project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f'Grid project ({project_name}) not found')

    try:
        project_meta_file = project_dir / settings.GRID_PROJECT_META_FILE_NAME
        with open(project_meta_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')
    
    project_meta = ProjectMeta(**data)
    project_path = Path(settings.GRID_PROJECT_DIR, project_meta.name)

    # Check if schema is valid
    schema_file_path = Path(settings.GRID_SCHEMA_DIR) / f'{project_meta.schema_name}.json'
    if not schema_file_path.exists():
        raise FileNotFoundError(f'Schema file {schema_file_path} does not exist')

    # Check if the patch directory already exists
    patch_dir = project_dir / data.name
    if patch_dir.exists():
        return BaseResponse(
            success=False,
            message='Grid patch already exists. Please use a different name.'
        )

    # Write the patch meta information to a file
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_meta_file = patch_dir / settings.GRID_PATCH_META_FILE_NAME
    try:
        with open(patch_meta_file, 'w') as f:
            f.write(data.model_dump_json(indent=4))
            node_key = f'root/projects/{project_name}/{data.name}'
            BT.instance.mount_node(
                'topo', node_key,
                {
                    'temp': settings.GRID_PATCH_TEMP,
                    'schema_file_path': str(schema_file_path),
                    'grid_project_path': str(project_path / data.name),
                    'meta_file_name': settings.GRID_PATCH_META_FILE_NAME,
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save grid patch meta information: {str(e)}')

    return BaseResponse(
        success=True,
        message='Grid patch created successfully'
    )

@router.put('/{project_name}/{patch_name}', response_model=BaseResponse)
def update_patch(project_name: str, patch_name: str, data: PatchMeta):
    """
    Description
    --
    Update a specific patch by new meta information.
    """

    # Check if the patch directory exists
    project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
    patch_dir = project_dir / patch_name
    if not patch_dir.exists():
        raise HTTPException(status_code=404, detail=f'Patch ({patch_name}) belonging to project ({project_name}) not found')

    # Write the updated patch meta information to a file
    patch_meta_file = patch_dir / settings.GRID_PATCH_META_FILE_NAME
    try:
        with open(patch_meta_file, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update grid patch meta information: {str(e)}')

    return BaseResponse(
        success=True,
        message='Grid patch updated successfully'
    )

@router.delete('/{project_name}/{patch_name}', response_model=BaseResponse)
def delete_patch(project_name: str, patch_name: str):
    """
    Description
    --
    Delete a patch by specific names of project and patch.
    """

    # Check if the patch directory exists
    project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
    patch_dir = project_dir / patch_name
    if not patch_dir.exists():
        raise HTTPException(status_code=404, detail='Patch not found')

    # Delete the patch directory
    try:
        for item in patch_dir.iterdir():
            item.unlink()
        patch_dir.rmdir()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete patch ({patch_name}) belonging to project ({project_name}): {str(e)}')

    return BaseResponse(
        success=True,
        message='Patch deleted successfully'
    )

@router.get('/feature/{project_name}/{patch_name}', response_model=BaseResponse)
def set_patch_feature(project_name: str, patch_name: str):
    """
    Description
    --
    Set a specific patch as the current crm server.
    """
    
    # Check if the patch directory exists
    project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
    patch_dir = project_dir / patch_name
    if not patch_dir.exists():
        raise HTTPException(status_code=404, detail=f'Grid patch ({patch_name}) belonging to project ({project_name}) not found')

    try:
        project_meta_file = project_dir / settings.GRID_PROJECT_META_FILE_NAME
        with open(project_meta_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')
    
    project_meta = ProjectMeta(**data)
    try:
        set_current_feature(project_meta, patch_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set patch as the current resource: {str(e)}')
    return BaseResponse(
        success=True,
        message='Grid patch set successfully'
    )