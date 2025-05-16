import json
import logging
import numpy as np
import c_two as cc
from osgeo import ogr, osr
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

@router.get('/activate-info', response_class=Response, response_description='Returns active grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def activate_grid_infos():
    
    with cc.compo.runtime.connect_crm(settings.TCP_ADDRESS, IGrid) as grid_interface:
        levels, global_ids = grid_interface.get_active_grid_infos()
        
        grid_infos = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
        
        return Response(
            content=grid_infos.combine_bytes(),
            media_type='application/octet-stream'
        )

@router.post('/subdivide', response_class=Response, response_description='Returns subdivided grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def subdivide_grids(grid_info: grid.MultiGridInfo):
    
    with cc.compo.runtime.connect_crm(settings.TCP_ADDRESS, IGrid) as grid_interface:
        keys = grid_interface.subdivide_grids(grid_info.levels, grid_info.global_ids)

        levels, global_ids = _keys_to_levels_global_ids(keys)
        subdivide_info = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
        
        return Response(
            content=subdivide_info.combine_bytes(),
            media_type='application/octet-stream'
        )

@router.get('/subdivide/{feature_dir}', response_class=Response, response_description='Returns subdivided grid information in bytes. Format: [4 bytes for length, followed by level bytes, followed by padding bytes, followed by global id bytes]')
def subdivide_grids_by_feature(feature_dir: str):
    """
    Subdivide grids based on features from a .shp or .geojson file.
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
        # Get schema name from the current project
        current_project = settings.APP_CONTEXT['current_project']
        project_meta_file = Path(settings.PROJECT_DIR, current_project, settings.GRID_PROJECT_META_FILE_NAME)
        with open(project_meta_file, 'r') as f:
            data = json.load(f)
        project_meta = grid.ProjectMeta(**data)
        schema_name = project_meta.schema_name
        
        # Get EPSG code from the schema
        schema_file = Path(settings.SCHEMA_DIR, f'{schema_name}.json')
        with open(schema_file, 'r') as f:
            data = json.load(f)
        schema_meta = grid.ProjectSchema(**data)
        target_epsg = schema_meta.epsg
        
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
            
            # Create spatial reference transformation from the layer to the target EPSG
            coord_trans = None
            source_sr = layer.GetSpatialRef()
            if source_sr and target_sr:
                # Check if transformation is needed
                if not source_sr.IsSame(target_sr):
                    # Ensure source_sr also uses traditional GIS order for consistency if it's a geographic CRS
                    if source_sr.IsGeographic() and int(osr.GetPROJVersionMajor())  >= 3:
                        source_sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    coord_trans = osr.CreateCoordinateTransformation(source_sr, target_sr)
                    if coord_trans is None:
                        logging.error(f'Failed to create coordinate transformation from {source_sr.GetName()} to {target_sr.GetName()}')
                        raise HTTPException(status_code=500, detail=f'Coordinate transformation failed for {feature_dir}')
            elif not source_sr:
                logging.warning(f'Layer {i} in {feature_dir} has no spatial reference. Assuming it matches target EPSG:{target_epsg}.')
            
            # Iterate through features in the layer and extract geometries
            feature = layer.GetNextFeature()
            while feature:
                geom = feature.GetGeometryRef()
                if geom:
                    if coord_trans:
                        err = geom.Transform(coord_trans)
                        if err != ogr.OGRERR_NONE:
                            logging.error(f'Error transforming geometry: {err}')
                            feature.Destroy()  # destroy the feature that failed transformation
                            feature = layer.GetNextFeature()
                            continue
                    ogr_geometries.append(geom)
                ogr_features.append(feature)
                feature = layer.GetNextFeature()
        
        if not ogr_geometries:
            logging.warning(f'No geometries found or extracted from feature file: {feature_dir}')
            raise HTTPException(status_code=400, detail=f'No geometries found in feature file: {feature_dir}')
        
        # Step 3: Get grids that need to be subdivided (i.e., grid centers within the feature geometries)
        grids_to_subdivide_levels: list[int] = []
        grids_to_subdivide_global_ids: list[int] = []
        with cc.compo.runtime.connect_crm(settings.TCP_ADDRESS, IGrid) as grid_interface:
            # Get all active grids
            active_levels, active_global_ids = grid_interface.get_active_grid_infos()
            if not active_levels or not active_global_ids:
                logging.info(f'No active grids found to check against features from {feature_dir}')
                return Response(
                    content=grid.MultiGridInfo(levels=[], global_ids=[]).combine_bytes(),
                    media_type='application/octet-stream'
                )

            # Get grid centers
            centers: list[tuple[float, float]] = grid_interface.get_multi_grid_centers(active_levels, active_global_ids)

            # Check if centers are in the feature geometries
            for i, center_coord in enumerate(centers):
                center_point = ogr.Geometry(ogr.wkbPoint)
                center_point.AssignSpatialReference(target_sr)
                center_point.AddPoint(center_coord[0], center_coord[1]) # X, Y order

                for feature_geom in ogr_geometries:
                    if feature_geom.Intersects(center_point) or feature_geom.Contains(center_point):
                        grids_to_subdivide_levels.append(active_levels[i])
                        grids_to_subdivide_global_ids.append(active_global_ids[i])
                        break
                if center_point:
                    center_point.Destroy()

            if not grids_to_subdivide_levels:
                logging.info(f'No active grid centers found within the features from {feature_dir}')
                return Response(
                    content=grid.MultiGridInfo(levels=[], global_ids=[]).combine_bytes(),
                    media_type='application/octet-stream'
                )

            # Step 4: Subdivide the selected grids
            keys = grid_interface.subdivide_grids(grids_to_subdivide_levels, grids_to_subdivide_global_ids)

        levels, global_ids = _keys_to_levels_global_ids(keys)
        subdivide_info = grid.MultiGridInfo(levels=levels, global_ids=global_ids)
        
        return Response(
            content=subdivide_info.combine_bytes(),
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to subdivide grids by feature: {str(e)}') 
    finally:
        # Clean up
        for feature in ogr_features:
            if feature:
                feature.Destroy()
        ogr_features.clear()
        ogr_geometries.clear()
        
        if coord_trans:
            coord_trans = None
        if target_sr:
            target_sr = None
        if source_sr:
            source_sr = None
        if data_source:
            data_source = None

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

def _keys_to_levels_global_ids(keys: list[str | None]) -> tuple[list[int], list[int]]:
    """
    Convert grid keys to levels and global IDs.
    Args:
        keys (list[str | None]): List of grid keys in the format "level-global_id"
    Returns:
        tuple[list[int], list[int]]: Tuple of two lists - levels and global IDs
    """
    if not keys:
        return [], []

    levels: list[int] = []
    global_ids: list[int] = []
    for key in keys:
        if key is None:
            continue
        level, global_id = map(int, key.split('-'))
        levels.append(level)
        global_ids.append(global_id)
    return levels, global_ids
