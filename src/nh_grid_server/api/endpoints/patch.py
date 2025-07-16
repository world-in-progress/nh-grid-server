import c_two as cc
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ...schemas.grid import GridMeta
from ...schemas.patch import PatchMeta
from ...schemas.base import BaseResponse
from ...schemas.crm import CRMStatus
from icrms.itreeger import ReuseAction, CRMDuration
from ...core.bootstrapping_treeger import BT
from ...core.config import settings, APP_CONTEXT

# APIs for grid patch ################################################

router = APIRouter(prefix='/patch', tags=['patch-related apis'])

@router.get('/', response_model=CRMStatus)
def check_patch_ready():
    """
    Description
    --
    Check if the patch server is ready.
    """
    node_key = APP_CONTEXT['current_patch']
    if not node_key:
        raise HTTPException(status_code=404, detail='No patch is currently set')
    
    try:
        server_address = BT.instance.get_node_info(node_key).server_address
        flag = cc.rpc.Client.ping(server_address)

        return CRMStatus(
            status='ACTIVATED' if flag else 'DEACTIVATED',
            is_ready=flag
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to check CRM of patch {node_key}: {str(e)}')


@router.get('/{schema_name}/{patch_name}', response_model=BaseResponse)
def set_patch(schema_name: str, patch_name: str):
    """
    Description
    --
    Set a specific patch as the current crm server.
    """
    # Check if the patch directory exists
    grid_patch_path = Path(settings.GRID_SCHEMA_DIR, schema_name, 'patches', patch_name)
    if not grid_patch_path.exists():
        raise HTTPException(status_code=404, detail=f'Grid patch ({patch_name}) belonging to schema ({schema_name}) not found')

    try:
        node_key = f'root.topo.schemas.{schema_name}.patches.{patch_name}'
        APP_CONTEXT['current_patch'] = node_key
        BT.instance.activate_node(node_key, ReuseAction.REPLACE, CRMDuration.Much_Long)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set patch as the current resource: {str(e)}')
    return BaseResponse(
        success=True,
        message='Grid patch set successfully'
    )

@router.get('/{schema_name}/{patch_name}/meta', response_model=GridMeta)
def get_patch_meta(schema_name: str, patch_name: str):
    """
    Get grid meta information from a specific patch.
    """
    # Check if the patch directory exists
    grid_patch_path = Path(settings.GRID_SCHEMA_DIR, schema_name, 'patches', patch_name)
    if not grid_patch_path.exists():
        raise HTTPException(status_code=404, detail=f'Grid patch ({patch_name}) belonging to schema ({schema_name}) not found')
    
    try:
        return GridMeta.from_patch(schema_name, patch_name)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')

@router.post('/{schema_name}', response_model=BaseResponse)
def create_patch(schema_name: str, patch_data: PatchMeta):
    """
    Description
    --
    Create a patch belonging to a specified schema.
    """

    # Check if the schema directory exists
    grid_schema_path = Path(settings.GRID_SCHEMA_DIR, schema_name, 'schema.json')
    if not grid_schema_path.exists():
        raise HTTPException(status_code=404, detail=f'Grid schema ({schema_name}) not found')
    
    try:
        grid_patch_path = Path(settings.GRID_SCHEMA_DIR, schema_name, 'patches', patch_data.name)
        if grid_patch_path.exists():
            return BaseResponse(
                success=False,
                message='Grid patch already exists. Please use a different name.'
            )

        # Write the patch meta information to a file
        grid_patch_path.mkdir(parents=True, exist_ok=True)
        patch_meta_file = grid_patch_path / settings.GRID_PATCH_META_FILE_NAME
        
        with open(patch_meta_file, 'w') as f:
            f.write(patch_data.model_dump_json(indent=4))
            node_key = f'root.topo.schemas.{schema_name}.patches.{patch_data.name}'

            # Mount the patch node
            BT.instance.mount_node('patch', node_key,
                                   {
                                       'schema_file_path': str(grid_schema_path),
                                       'grid_patch_path': str(grid_patch_path),
                                   })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to create grid patch: {str(e)}')

    return BaseResponse(
        success=True,
        message='Grid patch created successfully'
    )

@router.put('/{schema_name}/{patch_name}', response_model=BaseResponse)
def update_patch(schema_name: str, patch_name: str, data: PatchMeta):
    """
    Description
    --
    Update a specific patch by new meta information.
    """

    # Check if the patch directory exists
    grid_patch_dir = Path(settings.GRID_SCHEMA_DIR, schema_name, 'patches', patch_name)
    if not grid_patch_dir.exists():
        raise HTTPException(status_code=404, detail=f'Patch ({patch_name}) belonging to schema ({schema_name}) not found')

    # Write the updated patch meta information to a file
    patch_meta_file = grid_patch_dir / settings.GRID_PATCH_META_FILE_NAME
    try:
        with open(patch_meta_file, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update grid patch meta information: {str(e)}')

    return BaseResponse(
        success=True,
        message='Grid patch updated successfully'
    )

@router.delete('/{schema_name}/{patch_name}', response_model=BaseResponse)
def delete_patch(schema_name: str, patch_name: str):
    """
    Description
    --
    Delete a patch by specific name of schema and patch.
    """

    # Check if the patch directory exists
    grid_patch_dir = Path(settings.GRID_SCHEMA_DIR, schema_name, 'patches', patch_name)
    if not grid_patch_dir.exists():
        raise HTTPException(status_code=404, detail='Patch not found')

    # Delete the patch directory
    try:
        for item in grid_patch_dir.iterdir():
            item.unlink()
        grid_patch_dir.rmdir()

        # Unmount the patch node
        node_key = f'root.topo.schemas.{schema_name}.patches.{patch_name}'
        BT.instance.unmount_node(node_key)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete patch ({patch_name}) belonging to schema ({schema_name}): {str(e)}')

    return BaseResponse(
        success=True,
        message='Patch deleted successfully'
    )