import logging
from pathlib import Path
from osgeo import ogr, osr
import multiprocessing as mp
from functools import partial
from fastapi import APIRouter, Response, HTTPException, Body

from ....schemas import grid, base
from ....core.bootstrapping_treeger import BT
from ....core.config import settings, APP_CONTEXT

from icrms.itopo import ITopo, GridSchema, TopoSaveInfo

# APIs for grid operations ################################################

router = APIRouter(prefix='/operation', tags=['grid / operation'])

@router.get('/meta/{project_name}/{patch_name}', response_model=grid.GridMeta)
def get_grid_meta(project_name: str, patch_name: str):
    """
    Get grid meta information for a specific patch
    """
    try:
        project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
        patch_dir = project_dir / patch_name
        if not project_dir.exists() or not patch_dir.exists():
            raise HTTPException(status_code=404, detail='Project or patch not found')

        return grid.GridMeta.from_patch(project_name, patch_name)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')

@router.get('/meta', response_model=grid.GridMeta)
def get_current_grid_meta():
    """
    Get grid meta information of the current patch
    """
    try:
        return grid.GridMeta.from_context()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')
    
@router.get('/activate-info', response_class=Response, response_description='Returns active grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def activate_grid_info():
    with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
        levels, global_ids = topo.get_active_grid_infos()
    grid_infos = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
    
    return Response(
        content=grid_infos.combine_bytes(),
        media_type='application/octet-stream'
    )

@router.get('/deleted-info', response_class=Response, response_description='Returns deleted grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def deleted_grid_infos():
    with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
        levels, global_ids = topo.get_deleted_grid_infos()
    grid_infos = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
    
    return Response(
        content=grid_infos.combine_bytes(),
        media_type='application/octet-stream'
    )

@router.post('/subdivide', response_class=Response, response_description='Returns subdivided grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def subdivide_grids(grid_info_bytes: bytes = Body(..., description='Grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')):
    grid_info = grid.MultiGridInfo.from_bytes(grid_info_bytes)
    with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
        levels, global_ids = topo.subdivide_grids(grid_info.levels, grid_info.global_ids)
    subdivide_info = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
    return Response(
        content=subdivide_info.combine_bytes(),
        media_type='application/octet-stream'
    )

@router.post('/merge', response_class=Response, response_description='Returns merged grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def merge_grids(grid_info_bytes: bytes = Body(..., description='Grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')):
    """
    Merge grids based on the provided grid information
    """
    grid_info = grid.MultiGridInfo.from_bytes(grid_info_bytes)
    with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
        levels, global_ids = topo.merge_multi_grids(grid_info.levels, grid_info.global_ids)
        merge_info = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
    return Response(
        content=merge_info.combine_bytes(),
        media_type='application/octet-stream'
    )
        
@router.post('/delete', response_model=base.BaseResponse)
def delete_grids(grid_info_bytes: bytes = Body(..., description='Grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')):
    """
    Delete grids based on the provided grid information
    """
    try:
        with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
            grid_info = grid.MultiGridInfo.from_bytes(grid_info_bytes)
            topo.delete_grids(grid_info.levels, grid_info.global_ids)
        
        return base.BaseResponse(
            success=True,
            message='Grids deleted successfully'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete grids: {str(e)}')

@router.post('/recover', response_model=base.BaseResponse)
def recover_grids(grid_info_bytes: bytes = Body(..., description='Grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')):
    """
    Recover grids based on the provided grid information
    """
    try:
        grid_info = grid.MultiGridInfo.from_bytes(grid_info_bytes)
        with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
            topo.recover_multi_grids(grid_info.levels, grid_info.global_ids)

        return base.BaseResponse(
            success=True,
            message='Grids recovered successfully'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to recover grids: {str(e)}')

@router.get('/pick', response_class=Response, response_description='Returns picked grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def pick_grids_by_feature(feature_dir: str):
    """
    Pick grids based on features from a .shp or .geojson file.
    The feature_dir parameter should be a path to the feature file accessible by the server.
    """
    # Validate the feature_dir parameter
    feature_file = Path(feature_dir)
    file_extension = feature_file.suffix.lower()
    if file_extension not in ['.shp', '.geojson']:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {file_extension}. Must be .shp or .geojson.')
    if not feature_file.exists() or not feature_file.is_file():
        raise HTTPException(status_code=404, detail=f'Feature file not found: {feature_dir}')

    try:
        # Step 1: Prepare target spatial reference
        with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
            schema: GridSchema = topo.get_schema()
        target_epsg: int = schema.epsg
        target_sr = osr.SpatialReference()
        target_sr.ImportFromEPSG(target_epsg)
        # Ensure axis order is as expected by WKT (typically X, Y or Lon, Lat)
        # For EPSG > 4000, it's often Lat, Lon. For WKT, it's usually Lon, Lat.
        # OGR/GDAL 3+ handles this better, but being explicit can help.
        if int(osr.GetPROJVersionMajor()) >= 3:
            target_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        # Step 2: Get all features from the file
        ogr_features = []
        ogr_geometries = []
        
        # Set up GDAL/OGR data source
        data_source = ogr.Open(str(feature_file))
        if data_source is None:
            logging.error(f'GDAL/OGR could not open feature file: {feature_dir}')
            raise HTTPException(status_code=500, detail=f'Could not open feature file: {feature_dir}')

        for i in range(data_source.GetLayerCount()):
            layer = data_source.GetLayer(i)
            if layer is None:
                logging.warning(f'Could not get layer {i} from {feature_dir}')
                continue
            
            # Check if the layer has a same spatial reference as the target EPSG
            source_sr = layer.GetSpatialRef()
            if source_sr and target_sr:
                if not source_sr.IsSame(target_sr):
                    raise HTTPException(status_code=500, detail=f'Provided feature file has different EPSG {source_sr.GetAttrValue("AUTHORITY", 1)} than the target EPSG: {target_epsg}')
            elif not source_sr:
                raise HTTPException(status_code=500, detail=f'Layer {i} in {feature_dir} has no spatial reference.')
            
            # Iterate through features in the layer and extract geometries
            feature = layer.GetNextFeature()
            while feature:
                geom = feature.GetGeometryRef()
                if geom:
                    ogr_geometries.append(geom)
                ogr_features.append(feature)
                feature = layer.GetNextFeature()
        
        if not ogr_geometries:
            logging.warning(f'No geometries found or extracted from feature file: {feature_dir}')
            raise HTTPException(status_code=400, detail=f'No geometries found in feature file: {feature_dir}')

        # Step 3: Get centers of all active grids
        with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
            active_levels, active_global_ids = topo.get_active_grid_infos()
            
            if not active_levels or not active_global_ids:
                logging.info(f'No active grids found to check against features from {feature_dir}')
                return Response(
                    content=grid.MultiGridInfo(levels=[], global_ids=[]).combine_bytes(),
                    media_type='application/octet-stream'
                )
            bboxes: list[float]  = topo.get_multi_grid_bboxes(active_levels, active_global_ids)

        # Step 3: Pick grids, centers of which are within the features, accelerate with multiprocessing
        picked_grids_levels: list[int] = []
        picked_grids_global_ids: list[int] = []
        
        # Batch processing
        n_cores = mp.cpu_count()
        total_grids = len(bboxes) // 4
        points_per_process = max(1000, total_grids // (n_cores * 2))
        batches = []
        for i in range(0, total_grids, points_per_process):
            end_idx = min(i + points_per_process, total_grids)
            batch_indices = list(range(i, end_idx))
            batch_bboxes = bboxes[i * 4:end_idx * 4]
            batch_levels = [active_levels[idx] for idx in batch_indices]
            batch_global_ids = [active_global_ids[idx] for idx in batch_indices]
            batches.append((batch_indices, batch_bboxes, batch_levels, batch_global_ids))
        
        geometry_wkts = [geom.ExportToWkt() for geom in ogr_geometries]    
        process_func = partial(_process_grid_batch, geometry_wkts=geometry_wkts)
        with mp.Pool(processes=min(n_cores, len(batches))) as pool:
            results = pool.map(process_func, batches)
            
            for batch_levels, batch_global_ids in results:
                picked_grids_levels.extend(batch_levels)
                picked_grids_global_ids.extend(batch_global_ids)

        if not picked_grids_levels:
            logging.info(f'No active grid centers found within the features from {feature_dir}')
            return Response(
                content=grid.MultiGridInfo(levels=[], global_ids=[]).combine_bytes(),
                media_type='application/octet-stream'
            )

        picked_info = grid.MultiGridInfo(levels=picked_grids_levels, global_ids=picked_grids_global_ids)
        return Response(
            content=picked_info.combine_bytes(),
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to pick grids by feature: {str(e)}')
    finally:
        # Clean up
        for feature in ogr_features:
            if feature:
                feature.Destroy()
        ogr_features.clear()
        ogr_geometries.clear()
        
        if target_sr:
            target_sr = None
        if source_sr:
            source_sr = None
        if data_source:
            data_source = None

@router.get('/save', response_model=base.BaseResponse)
def save_grids():
    """
    Save the current grid state to a file.
    """
    try:
        with BT.instance.connect(_get_current_patch_node(), ITopo) as topo:
            result: TopoSaveInfo = topo.save()
            logging.info(f'Grid saved successfully: {result}')
        return base.BaseResponse(
            success=result.success,
            message=result.message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save grid: {str(e)}')
    
# Helpers ##################################################

def _get_current_patch_node():
    return f'root/projects/{APP_CONTEXT.get("current_project")}/{APP_CONTEXT.get("current_patch")}'

def _process_grid_batch(batch_data, geometry_wkts):
    # _ is batch_indices
    _, batch_boxes, batch_levels, batch_global_ids = batch_data
    
    geometries = [ogr.CreateGeometryFromWkt(wkt) for wkt in geometry_wkts]
    picked_levels = []
    picked_global_ids = []
    
    box_geometry = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in range(len(batch_boxes) // 4):
        bbox = batch_boxes[i * 4:i * 4 + 4]
        minX, minY, maxX, maxY = bbox
        ring.Empty()
        ring.AddPoint(minX, minY)
        ring.AddPoint(maxX, minY)
        ring.AddPoint(maxX, maxY)
        ring.AddPoint(minX, maxY)
        ring.AddPoint(minX, minY) 
        
        box_geometry.Empty()
        box_geometry.AddGeometry(ring)
        
        for geom in geometries:
            if geom.Intersects(box_geometry) or geom.Contains(box_geometry):
                picked_levels.append(batch_levels[i])
                picked_global_ids.append(batch_global_ids[i])
                break
    
    ring.Destroy()
    box_geometry.Destroy()
    for geom in geometries:
        geom.Destroy()
    
    return picked_levels, picked_global_ids
