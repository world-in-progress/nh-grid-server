import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ...core.config import settings
from ...schemas.base import BaseResponse
from ...schemas.project import ProjectMeta
from ...core.bootstrapping_treeger import BT
from ...schemas.schema import GridSchema, ResponseWithGridSchema

# APIs for single grid schema ##################################################

router = APIRouter(prefix='/schema', tags=['schema-related apis'])

@router.get('/{name}', response_model=ResponseWithGridSchema)
def get_schema(name: str):
    """
    Description
    --
    Get a grid schema by name.
    """
    
    # Check if the schema file exists
    grid_schema_path = Path(settings.GRID_SCHEMA_DIR, name, 'schema.json')
    if not grid_schema_path.exists():
        raise HTTPException(status_code=404, detail='Grid schema not found')

    # Read the schema from the file
    try:
        with open(grid_schema_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read schema: {str(e)}')
    
    return ResponseWithGridSchema(
        grid_schema=GridSchema(**data)
    )

@router.post('/', response_model=BaseResponse)
def register_schema(data: GridSchema):
    """
    Description
    --
    Register a grid schema.
    """

    # Find if grid schema is existed
    grid_schema_path = Path(settings.GRID_SCHEMA_DIR, data.name, 'schema.json')
    if grid_schema_path.exists():
        return BaseResponse(
            success=False,
            message='Grid schema already exists. Please use a different name.'
        )
        
    # Write the schema to a file
    try:
        grid_schema_path.mkdir(parents=True, exist_ok=True)
        
        with open(grid_schema_path, 'w') as f:
            f.write(data.model_dump_json(indent=4))
            
        # Create resoruce folder for patches and grids
        patches_path = grid_schema_path.parent / 'patches'
        grids_path = grid_schema_path.parent / 'grids'
        patches_path.mkdir(exist_ok=True)
        grids_path.mkdir(exist_ok=True)
        
        # Mount scene nodes
        BT.instance.mount_node('schema', f'root.topo.schemas.{data.name}')
        BT.instance.mount_node('patches', f'root.topo.schemas.{data.name}.patches')
        BT.instance.mount_node('grids', f'root.topo.schemas.{data.name}.grids')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save grid schema: {str(e)}')
    return BaseResponse(
        success=True,
        message='Grid schema registered successfully'
    )

@router.put('/{name}', response_model=BaseResponse)
def update_schema(name: str, data: GridSchema):
    """
    Description
    --
    Update a grid schema by name.
    """
    
    # Check if the schema file exists
    grid_schema_path = Path(settings.GRID_SCHEMA_DIR, name, 'schema.json')
    if not grid_schema_path.exists():
        raise HTTPException(status_code=404, detail='Grid schema not found')

    # Write the updated schema to the file
    try:
        with open(grid_schema_path, 'w') as f:
            f.write(data.model_dump_json(indent=4))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update schema: {str(e)}')
    
    return BaseResponse(
        success=True,
        message='Project schema updated successfully'
    )

@router.delete('/{name}', response_model=BaseResponse)
def delete_schema(name: str):
    """
    Description
    --
    Delete a grid schema by name.
    """
    # Get Schema node key
    node_key = f'root.topo.schemas.{name}'
    
    # Check if the schema file exists
    grid_schema_path = Path(settings.GRID_SCHEMA_DIR, name, 'schema.json')
    if not grid_schema_path.exists():
        raise HTTPException(status_code=404, detail='Grid schema not found')
    
    try:
        # TODO: Delete all patches and grids under this schema and unmount them
        patches_path = grid_schema_path.parent / 'patches'
        grids_path = grid_schema_path.parent / 'grids'
        if patches_path.exists():
            for patch in patches_path.glob('*'):
                if patch.is_dir():
                    patch.rmdir()
                    BT.instance.unmount_node(f'root.topo.schemas.{name}.patches.{patch.name}')
                    
            patches_path.rmdir()
            BT.instance.unmount_node(f'root.topo.schemas.{name}.patches')

        if grids_path.exists():
            for grid in grids_path.glob('*'):
                if grid.is_dir():
                    grid.rmdir()
                    BT.instance.unmount_node(f'root.topo.schemas.{name}.grids.{grid.name}')
                    
            grids_path.rmdir()
            BT.instance.unmount_node(f'root.topo.schemas.{name}.grids')
                    
        grid_schema_path.unlink()
        grid_schema_path.parent.rmdir()
        BT.instance.unmount_node(node_key)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete schema: {str(e)}')
    
    return BaseResponse(
        success=True,
        message='Grid schema deleted successfully'
    )