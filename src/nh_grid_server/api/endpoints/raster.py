import io
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Body

from icrms.iraster import IRaster
from icrms.ifeature import IFeature
from ...schemas.base import BaseResponse
from ...core.bootstrapping_treeger import BT
from icrms.itreeger import ReuseAction, CRMDuration
from ...schemas.raster import CreateRasterBody, UpdateByFeatureBody, GetCogTifResponse, SamplingResponse, GetMetadataResponse

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
        node_key = f'root.{body.type}s.{body.name}'
        BT.instance.mount_node(body.type, node_key, body.model_dump())
        return BaseResponse(
            success=True,
            message=node_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to create raster: {str(e)}')

@router.get('/cog_tif/{node_key}', response_model=GetCogTifResponse)
def get_cog_tif(node_key: str):
    """
    Description
    --
    Get the COG TIFF.
    """
    try:
        with BT.instance.connect(node_key, IRaster, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as raster:
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

@router.post('/update_by_features/{node_key}', response_model=BaseResponse)
def update_by_features(node_key: str, body: UpdateByFeatureBody = Body(..., description='Update raster by features')):
    """
    Description
    --
    Update raster data based on a list of GeoJSON features with operations.
    
    Parameters:
    - node_key: Key of the raster node to update
    - body: Contains a list of update operations, each with:
      - feature_node_key: Key to retrieve the GeoJSON feature
      - operation: Operation type ('set', 'add', 'subtract', 'max_fill')
      - value: Value to use in the operation (default is 0.0)
    
    The operations are applied sequentially to the TIF file, and COG is regenerated only once at the end.
    """
    try:
        if not body.updates:
            raise HTTPException(status_code=400, detail='No update operations provided')
        
        # 准备feature操作列表
        feature_operations = []
        
        for update_item in body.updates:
            # 从feature_node_key获取feature数据
            try:
                with BT.instance.connect(update_item.feature_node_key, IFeature, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as feature_node:
                    feature_json = feature_node.get_feature_json_computation()  # 获取feature的GeoJSON数据
                    
                    feature_operations.append({
                        'feature': feature_json,
                        'operation': update_item.operation,
                        'value': update_item.value
                    })
            except Exception as fe:
                logger.error(f'Failed to retrieve feature from {update_item.feature_node_key}: {str(fe)}')
                raise HTTPException(status_code=404, detail=f'Feature not found: {update_item.feature_node_key}')
        
        # 执行批量更新
        with BT.instance.connect(node_key, IRaster, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as raster:
            updated_path = raster.update_by_features(feature_operations)
            if not updated_path:
                raise HTTPException(status_code=500, detail='Failed to update raster')
            
            return BaseResponse(
                success=True,
                message=f"Raster updated successfully with {len(feature_operations)} operations."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update raster by features: {str(e)}')

@router.get('/sampling/{node_key}/{x}/{y}', response_model=SamplingResponse)
def get_raster_sampling(node_key: str, x: float, y: float):
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
        with BT.instance.connect(node_key, IRaster, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as raster:
            value = raster.sampling(x, y)
            return SamplingResponse(
                success=True,
                message="Raster sampling retrieved successfully.",
                data={"x": x, "y": y, "value": value}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get raster sampling: {str(e)}')

@router.get('/tile/{node_key}/{encoding}/{timestamp}/{z}/{x}/{y}.png')
def get_raster_tile_png(node_key: str, x: int, y: int, z: int, timestamp: int, encoding: str = "terrainrgb"):
    """
    Description
    --
    Get raster tile as PNG image using COG optimization.
    
    Parameters:
    - node_key: Key of the raster node
    - x: Tile X coordinate  
    - y: Tile Y coordinate
    - z: Zoom level
    - timestamp: Timestamp for the raster data
    - encoding: Encoding format ("terrainrgb" or "uint8"), default is "terrainrgb"
    """
    try:
        with BT.instance.connect(node_key, IRaster, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as raster:
            png_data = raster.get_tile_png(x, y, z, encoding)
            if not png_data:
                raise HTTPException(status_code=404, detail='Tile not found')
            
            return StreamingResponse(
                io.BytesIO(png_data),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=3600"}
            )
            # return Response(content=png_data, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get raster tile: {str(e)}')

@router.get('/metadata/{node_key}', response_model=GetMetadataResponse)
def get_raster_metadata(node_key: str):
    """
    Description
    --
    Get raster metadata information.
    
    Parameters:
    - raster_name: Name of the raster
    """
    try:
        with BT.instance.connect(node_key, IRaster, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as raster:
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
    
@router.delete('/{node_key}', response_model=BaseResponse)
def delete_raster(node_key: str):
    """
    Description
    --
    Delete a raster.
    """
    try:
        with BT.instance.connect(node_key, IRaster, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as raster:
            delete_info = raster.delete_raster()
        if delete_info.get('success', False):
            BT.instance.unmount_node(node_key)
        return BaseResponse(
            success=delete_info.get('success', False),
            message=delete_info.get('message', '')
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to delete feature: {str(e)}'
        )