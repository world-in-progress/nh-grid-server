import numpy as np
import c_two as cc
from pathlib import Path
from fastapi import APIRouter, Response, HTTPException

from icrms.igrid import IGrid
from ....schemas import grid
from ....compos import grid_comp
from ....core.config import settings

# APIs for grid operations ################################################

router = APIRouter(prefix='/operation', tags=['grid / operation'])

@router.get('/meta', response_model=grid.GridMeta)
def get_grid_meta():
    """
    Get grid meta information for a specific subproject.
    """
    try:
        # project_dir = Path(settings.PROJECT_DIR, project_name)
        # subproject_dir = project_dir / subproject_name
        # if not subproject_dir.exists() or not project_dir.exists():
        #     raise HTTPException(status_code=404, detail=f'Grid subproject ({subproject_name}) belonging to project ({project_name}) not found')
        
        return grid.GridMeta.from_context()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')

@router.get('/activate-info', response_class=Response, response_description='Returns active grid information in bytes. Formart: [4 bytes for length, followed by level bytes, followed by global id bytes]')
def activate_grid_infos():
    
    with cc.compo.runtime.connect_crm(settings.TCP_ADDRESS, IGrid) as grid_interface:
        levels, global_ids = grid_interface.get_active_grid_infos()
        
        level_bytes = np.array(levels, dtype=np.int8).tobytes()
        global_id_bytes = np.array(global_ids, dtype=np.int32).tobytes()
        
        level_length = len(level_bytes).to_bytes(4, byteorder='little')
        combined_data = level_length + level_bytes + global_id_bytes
        
        return Response(
            content=combined_data,
            media_type='application/octet-stream'
        )
        