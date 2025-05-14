import logging
import numpy as np
import c_two as cc
from pathlib import Path
from fastapi import APIRouter, Response, HTTPException

from icrms.igrid import IGrid
from ....schemas import grid, base
from ....core.config import settings

# APIs for grid operations ################################################

router = APIRouter(prefix='/operation', tags=['grid / operation'])

@router.get('/meta', response_model=grid.GridMeta)
def get_current_grid_meta():
    """
    Get grid meta information of the current subproject
    """
    try:
        return grid.GridMeta.from_context()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')

@router.get('/meta/{project_name}/{subproject_name}', response_model=grid.GridMeta)
def get_grid_meta(project_name: str, subproject_name: str):
    """
    Get grid meta information for a specific subproject
    """

    try:
        project_dir = Path(settings.PROJECT_DIR, project_name)
        subproject_dir = project_dir / subproject_name
        if not project_dir.exists() or not subproject_dir.exists():
            raise HTTPException(status_code=404, detail='Project or subproject not found')

        return grid.GridMeta.from_subproject(project_name, subproject_name)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')

@router.get('/activate-info', response_class=Response, response_description='Returns active grid information in bytes. Formart: [4 bytes for length, followed by level bytes, followed by global id bytes]')
def activate_grid_infos():
    
    with cc.compo.runtime.connect_crm(settings.TCP_ADDRESS, IGrid) as grid_interface:
        levels, global_ids = grid_interface.get_active_grid_infos()
        
        level_bytes = np.array(levels, dtype=np.int8).tobytes()
        global_id_bytes = np.array(global_ids, dtype=np.int32).tobytes()
        
        level_length = len(level_bytes).to_bytes(4, byteorder='little')
        padding_size = (4 - (len(level_length) + len(level_bytes)) % 4) % 4
        padding = b'\x00' * padding_size
        
        combined_data = level_length + level_bytes + padding + global_id_bytes
        logging.debug(f'Activate grid info: {len(levels)}, {len(global_ids)}, {len(combined_data)}')
        
        return Response(
            content=combined_data,
            media_type='application/octet-stream'
        )

@router.post('/subdivide', response_class=Response, response_description='Returns subdivided grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by global id bytes]')
def subdivide_grids(grid_info: grid.MultiGridInfo):
    
    with cc.compo.runtime.connect_crm(settings.TCP_ADDRESS, IGrid) as grid_interface:
        keys = grid_interface.subdivide_grids(grid_info.levels, grid_info.global_ids)

        levels, global_ids = _keys_to_levels_global_ids(keys)
        subdivide_info = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
        
        return Response(
            content=subdivide_info.combine_bytes(),
            media_type='application/octet-stream'
        )

@router.post('/delete', response_model=base.BaseResponse)
def delete_grids(grid_info: grid.MultiGridInfo):
    """
    Delete grids based on the provided grid information
    """
    
    try:
        with cc.compo.runtime.connect_crm(settings.TCP_ADDRESS, IGrid) as grid_interface:
            grid_interface.delete_grids(grid_info.levels, grid_info.global_ids)
            
            return base.BaseResponse(
                success=True,
                message='Grids deleted successfully'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete grids: {str(e)}')

# Helpers ##################################################

def _keys_to_levels_global_ids(keys: list[str]) -> tuple[list[int], list[int]]:
    """
    Convert grid keys to levels and global IDs
    Args:
        keys (list[str]): List of grid keys in the format "level-global_id"
    Returns:
        tuple[list[int], list[int]]: Tuple of two lists - levels and global IDs
    """
    levels: list[int] = []
    global_ids: list[int] = []
    for key in keys:
        level, global_id = map(int, key.split('-'))
        levels.append(level)
        global_ids.append(global_id)
    return levels, global_ids