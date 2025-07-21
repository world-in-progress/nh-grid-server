import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ...core.config import settings
from ...schemas.base import BaseResponse
from ...schemas.grids import PatchNodeInfos
from ...core.bootstrapping_treeger import BT, CRMDuration

from icrms.igrid import IGrid

# APIs for grid patch ################################################

router = APIRouter(prefix='/grids', tags=['grids-related apis'])

@router.post('/{schema_name}/{grid_name}')
def create_grid(schema_name: str, grid_name: str, grid_patches: PatchNodeInfos):
    grid_path = Path(settings.GRID_SCHEMA_DIR, schema_name, 'grids', grid_name)
    # if grid_path.exists():
    #     raise HTTPException(status_code=400, detail=f'Grid {grid_name} already exists in schema {schema_name}')

    try:
        # Create the grid directory
        grid_path.mkdir(parents=True, exist_ok=True)
        
        # Save the patches information
        patches_file = grid_path / 'patches.json'
        with open(patches_file, 'w') as f:
            f.write(grid_patches.model_dump_json(indent=4))
            
        # Mount the grid node in Treeger
        node_key = _get_grid_node_key(schema_name, grid_name)
        BT.instance.mount_node('grid', node_key, {
            'schema_path': str(Path(settings.GRID_SCHEMA_DIR, schema_name, 'schema.json')),
            'workspace': str(grid_path)
        })
        
        # Connect to the grid node and merge patches
        with BT.instance.connect(node_key, IGrid, CRMDuration.Once) as grid:
            grid.merge()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to create grid: {str(e)}')

@router.delete('/{schema_name}/{grid_name}', response_model=BaseResponse)
def delete_grid(schema_name: str, grid_name: str):
    grid_path = Path(settings.GRID_SCHEMA_DIR, schema_name, 'grids', grid_name)
    if not grid_path.exists():
        raise HTTPException(status_code=404, detail=f'Grid {grid_name} not found in schema {schema_name}')
    try:
        grid_node_key = _get_grid_node_key(schema_name, grid_name)
        BT.instance.unmount_node(grid_node_key)
        shutil.rmtree(grid_path)

        return BaseResponse(success=True, message=f'Grid {grid_name} deleted successfully')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete grid: {str(e)}')

# Helpers ##################################################

def _get_grid_node_key(schema_name: str, grid_name: str) -> str:
    return f'root.topo.schemas.{schema_name}.grids.{grid_name}'