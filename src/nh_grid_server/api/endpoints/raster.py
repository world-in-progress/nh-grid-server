from fastapi import APIRouter
from fastapi import APIRouter, HTTPException, Body, Response
from fastapi.responses import StreamingResponse
import logging
import io
from typing import Optional
from ...core.bootstrapping_treeger import BT
from ...schemas.base import BaseResponse
from ...schemas.raster import CreateRasterBody, UpdateByFeatureBody, GetCogTifResponse, SamplingResponse, GetMetadataResponse
from icrms.iraster import IRaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/raster', tags=['raster / operation'])

@router.post('/create', response_model=BaseResponse)
def create_raster(body: CreateRasterBody=Body(..., description='create raster')):
    """
    Description
    --
    Create a raster.
    """
    try:
        node_key = f'root.dems.{body.name}'
        BT.instance.mount_node("dem", node_key, body.model_dump())
        return BaseResponse(
            success=True,
            message=node_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set patch as the current resource: {str(e)}')

@router.get('/cog_tif/{raster_name}', response_model=GetCogTifResponse)
def get_cog_tif(raster_name: str):
    """
    Description
    --
    Get the COG TIFF.
    """
    try:
        node_key = f'root.dems.{raster_name}'
        with BT.instance.connect(node_key, IRaster) as raster:
            cog_tif = raster.get_cog_tif()
            if not cog_tif:
                raise HTTPException(status_code=404, detail='COG TIFF not found')
            return GetCogTifResponse(
                success=True,
                message="COG TIFF retrieved successfully.",
                data={"cog_tif": cog_tif}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to retrieve COG TIFF: {str(e)}')

@router.post('/update_by_feature/{raster_name}', response_model=BaseResponse)
def update_raster_by_feature(raster_name: str, body: UpdateByFeatureBody = Body(..., description='Update raster by feature')):
    """
    Description
    --
    Update raster data based on a GeoJSON feature.
    
    Parameters:
    - raster_name: Name of the raster to update
    - feature: GeoJSON feature object
    - operation: Operation type ('set', 'add', 'subtract'), default is 'set'
    - value: Value to use in the operation, default is 0.0
    """
    try:
        node_key = f'root.dems.{raster_name}'
        with BT.instance.connect(node_key, IRaster) as raster:
            updated_path = raster.update_by_feature(body.feature, body.operation, body.value)
            if not updated_path:
                raise HTTPException(status_code=500, detail='Failed to update raster')
            return BaseResponse(
                success=True,
                message="Raster updated successfully by feature.",
                data={"updated_raster_path": updated_path}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update raster by feature: {str(e)}')

@router.get('/sampling/{raster_name}/{x}/{y}', response_model=SamplingResponse)
def get_raster_sampling(raster_name: str, x: float, y: float):
    """
    Description
    --
    Get raster sampling value at specified coordinates.
    
    Parameters:
    - raster_name: Name of the raster
    - x: X coordinate
    - y: Y coordinate
    - src_crs: Source coordinate reference system (e.g., 'EPSG:4326', 'EPSG:3857')
              If None, assumes coordinates are in the same CRS as the raster
    """
    try:
        node_key = f'root.dems.{raster_name}'
        with BT.instance.connect(node_key, IRaster) as raster:
            value = raster.sampling(x, y)
            return SamplingResponse(
                success=True,
                message="Raster sampling retrieved successfully.",
                data={"x": x, "y": y, "value": value}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get raster sampling: {str(e)}')

@router.get('/tile/{raster_name}/{z}/{x}/{y}.png')
def get_raster_tile_png(raster_name: str, x: int, y: int, z: int):
    """
    Description
    --
    Get raster tile as PNG image using COG optimization.
    
    Parameters:
    - raster_name: Name of the raster
    - x: Tile X coordinate  
    - y: Tile Y coordinate
    - z: Zoom level
    """
    try:
        node_key = f'root.dems.{raster_name}'
        with BT.instance.connect(node_key, IRaster) as raster:
            png_data = raster.get_tile_png(x, y, z)
            if not png_data:
                raise HTTPException(status_code=404, detail='Tile not found')
            
            # return StreamingResponse(
            #     io.BytesIO(png_data),
            #     media_type="image/png",
            #     headers={"Cache-Control": "public, max-age=3600"}
            # )
            return Response(content=png_data, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get raster tile: {str(e)}')

@router.get('/metadata/{raster_name}', response_model=GetMetadataResponse)
def get_raster_metadata(raster_name: str):
    """
    Description
    --
    Get raster metadata information.
    
    Parameters:
    - raster_name: Name of the raster
    """
    try:
        node_key = f'root.dems.{raster_name}'
        with BT.instance.connect(node_key, IRaster) as raster:
            metadata = raster.get_metadata()
            if not metadata:
                raise HTTPException(status_code=404, detail='Metadata not found')
            return GetMetadataResponse(
                success=True,
                message="Raster metadata retrieved successfully.",
                data=metadata
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get raster metadata: {str(e)}')
    
