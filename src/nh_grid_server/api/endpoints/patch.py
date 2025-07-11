from pathlib import Path
from fastapi import APIRouter, HTTPException

from ...core.config import settings
from ...schemas.base import BaseResponse
from ...schemas.project import PatchMeta
from ...core.bootstrapping_treeger import BT

# APIs for grid patch ################################################

router = APIRouter(prefix='/patch')

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
            BT.instance.mount_node('patch', node_key)
            
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
        node_key = f'root/schemas/{schema_name}/patches/{patch_name}'
        BT.instance.unmount_node(node_key)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete patch ({patch_name}) belonging to schema ({schema_name}): {str(e)}')

    return BaseResponse(
        success=True,
        message='Patch deleted successfully'
    )