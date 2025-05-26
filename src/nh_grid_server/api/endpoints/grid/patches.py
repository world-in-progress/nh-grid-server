import json
import c_two as cc
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ....core.config import settings
from ....schemas.project import PatchMeta, ResponseWithPatchMetas

# APIs for multi grid patches ################################################

router = APIRouter(prefix='/patches', tags=['grid / patches'])

@router.get('/{project_name}', response_model=ResponseWithPatchMetas)
def get_multi_patch_meta(project_name: str):
    """
    Description
    --
    Get all meta information of patches belonging to a specified project.
    """

    # Check if the project directory exists
    project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f'Grid project ({project_name}) not found')

    # Get all patch directories
    patch_dirs = list(project_dir.glob('*'))
    patch_meta_files = [ patch_dir / settings.GRID_PATCH_META_FILE_NAME for patch_dir in patch_dirs if patch_dir.is_dir() ]
    patch_metas = []
    for file in patch_meta_files:
        with open(file, 'r') as f:
            data = json.load(f)
            patch_metas.append(PatchMeta(**data))

    # Sort patch meta information: first by starred (True first), then alphabetically by name
    patch_metas.sort(key=lambda meta: (not meta.starred, meta.name.lower()))
    return ResponseWithPatchMetas(
        patch_metas=patch_metas
    )
