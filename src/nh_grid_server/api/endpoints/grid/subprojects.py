import json
import c_two as cc
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ....core.config import settings
from ....schemas.project import SubprojectMeta, ResponseWithSubprojectMetas

# APIs for multi grid subproject ################################################

router = APIRouter(prefix='/subprojects', tags=['grid / subprojects'])

@router.get('/{project_name}', response_model=ResponseWithSubprojectMetas)
def get_multi_subproject_meta(project_name: str):
    """
    Description
    --
    Get all meta information of subprojects belonging to a specified project.
    """
    
    # Check if the project directory exists
    project_dir = Path(settings.PROJECT_DIR, project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f'Grid project ({project_name}) not found')
    
    # Get all subproject directories
    subproject_dirs = list(project_dir.glob('*'))
    subproject_meta_files = [ subproject_dir / settings.GRID_SUBPROJECT_META_FILE_NAME for subproject_dir in subproject_dirs if subproject_dir.is_dir() ]
    subproject_metas = []
    for file in subproject_meta_files:
        with open(file, 'r') as f:
            data = json.load(f)
            subproject_metas.append(SubprojectMeta(**data))
    
    # Sort subproject meta information: first by starred (True first), then alphabetically by name
    subproject_metas.sort(key=lambda meta: (not meta.starred, meta.name.lower()))
    return ResponseWithSubprojectMetas(
        subproject_metas=subproject_metas
    )
