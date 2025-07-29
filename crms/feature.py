import os
import json
import zipfile
import shutil
import c_two as cc
from typing import Any
from pathlib import Path
from osgeo import ogr, osr
from icrms.ifeature import IFeature
from src.nh_grid_server.core.config import settings
from src.nh_grid_server.schemas.feature import UpdateFeatureBody

import logging
logger = logging.getLogger(__name__)

DEFAULT_EPSG = "4326"  # Default EPSG code for GeoJSON storage

@cc.iicrm
class Feature(IFeature):
    """
    CRM
    =
    The Feature Resource.  
    Feature is a feature file that can be uploaded to the resource pool.  
    """
    def __init__(self, name: str, type: str, color: str, epsg: str):
        """Method to initialize Feature

        Args:
            feature_path (str): path to the feature file
        """
        self.name = name
        self.type = type
        self.color = color
        self.epsg = epsg

        self.path = Path(f'{settings.FEATURE_DIR}{self.name}')
        self.path.mkdir(parents=True, exist_ok=True)

    def save_feature(self, feature_json) -> dict[str, bool | str]:
        feature_path = self.path / f'{self.name}.geojson'
        try:
            # Ensure the feature_json is in FeatureCollection format
            feature_collection = self._ensure_feature_collection(feature_json)
            
            with open(feature_path, 'w', encoding='utf-8') as f:
                json.dump(feature_collection, f, ensure_ascii=False, indent=2)
            return {
                'success': True,
                'message': 'Feature saved successfully as FeatureCollection'
            }
        except Exception as e:
            logger.error(f'Failed to save feature: {str(e)}')
            return {
                'success': False,
                'message': str(e)
            }

    def save_uploaded_feature(self, file_path: str, file_type: str) -> dict[str, bool | str]:
        feature_path = self.path / f'{self.name}.geojson'
        try:
            if file_type == 'shp':
                feature_json, source_epsg = self._shp_to_geojson_with_epsg(file_path)
                # Shapefile conversion already returns FeatureCollection
            elif file_type == 'geojson':
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                # Ensure uploaded GeoJSON is in FeatureCollection format
                feature_json = self._ensure_feature_collection(raw_data)
                source_epsg = self._detect_geojson_epsg(feature_json)
            else:
                raise ValueError(f'Unsupported file type: {file_type}')
            
            # Convert to EPSG:4326 for storage if needed
            if source_epsg and source_epsg != DEFAULT_EPSG:
                logger.info(f'Converting feature from EPSG:{source_epsg} to EPSG:{DEFAULT_EPSG} for storage')
                feature_json = self._transform_geojson_coordinates(feature_json, source_epsg, DEFAULT_EPSG)
            
            with open(feature_path, 'w', encoding='utf-8') as f:
                json.dump(feature_json, f, ensure_ascii=False, indent=2)
            return {
                'success': True,
                'message': f'Feature saved successfully as FeatureCollection (converted from EPSG:{source_epsg} to EPSG:{DEFAULT_EPSG})' if source_epsg and source_epsg != DEFAULT_EPSG else 'Feature saved successfully as FeatureCollection'
            }
        except Exception as e:
            logger.error(f'Failed to save feature: {str(e)}')
            return {
                'success': False,
                'message': str(e)
            }

    def get_feature(self) -> dict[str, Any]:
        """
        Get feature json
        """
        file_path = os.path.join(self.path, self.name + '.geojson')
        with open(file_path, 'r', encoding='utf-8') as f:
            feature_json = json.load(f)
        return {
            'name': self.name,
            'type': self.type,
            'color': self.color,
            'epsg': self.epsg,
            'feature_json': feature_json
        }

    def get_feature_json_computation(self) -> dict[str, Any]:
        """
        Get feature json for computation
        Transform coordinates from EPSG:4326 (storage CRS) to self.epsg (computation CRS)
        """
        file_path = os.path.join(self.path, self.name + '.geojson')
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # If target EPSG is the same as storage EPSG (4326), return as is
        if self.epsg == DEFAULT_EPSG:
            logger.info('No transformation needed, returning GeoJSON as is')
            return geojson_data
        logger.info(f'Transforming GeoJSON coordinates from EPSG:4326 to EPSG:{self.epsg}')
        # Transform coordinates from 4326 to target EPSG
        return self._transform_geojson_coordinates(geojson_data, DEFAULT_EPSG, self.epsg)

    def update_feature(self, update_body: UpdateFeatureBody) -> dict[str, bool | str]:
        """
        Update feature
        """
        self.name = update_body.name
        self.type = update_body.type
        self.color = update_body.color
        self.epsg = update_body.epsg
        feature_json = update_body.feature_json
        # 保存更新后的feature_json
        file_path = os.path.join(self.path, self.name + '.geojson')
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(feature_json, f, ensure_ascii=False, indent=2)
            return {
                'success': True,
                'message': 'Feature updated successfully',
            }
        except Exception as e:
            logger.error(f'Failed to update feature: {str(e)}')
            return {
                'success': False,
                'message': str(e)
            }
        
    def delete_feature(self) -> dict[str, bool | str]:
        """
        Delete feature
        """
        if os.path.exists(self.path):
            try:
                # Use shutil.rmtree to remove directory and all its contents
                shutil.rmtree(self.path)
                return {
                    'success': True,
                    'message': 'Feature deleted successfully',
                }
            except PermissionError as e:
                logger.error(f'Permission denied when deleting feature: {str(e)}')
                return {
                    'success': False,
                    'message': f'Permission denied: {str(e)}',
                }
            except Exception as e:
                logger.error(f'Failed to delete feature: {str(e)}')
                return {
                    'success': False,
                    'message': f'Failed to delete feature: {str(e)}',
                }
        else:
            return {
                'success': False,   
                'message': 'Feature not found',
            }
 
    def _shp_to_geojson(self, shp_path):
        # 如果输入是zip文件，先解压
        if shp_path.endswith('.zip'):
            shp_path = self._unzip_and_get_shp(shp_path)
        # 打开 shp 文件
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(shp_path, 0)  # 0 means read-only

        if dataSource is None:
            raise FileNotFoundError(f"无法打开 {shp_path}")

        layer = dataSource.GetLayer()

        # 构造 GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        # 读取.prj文件（坐标参考）
        prj_path = os.path.splitext(shp_path)[0] + '.prj'
        crs_wkt = None
        if os.path.exists(prj_path):
            with open(prj_path, 'r', encoding='utf-8') as prj_file:
                crs_wkt = prj_file.read().strip()
        if crs_wkt:
            geojson["crs_wkt"] = crs_wkt

        # 遍历所有要素
        for feature in layer:
            geom = feature.GetGeometryRef()
            # 转成 GeoJSON geometry 字典
            geom_json = json.loads(geom.ExportToJson())

            # 读取属性表
            properties = {}
            feature_defn = layer.GetLayerDefn()
            for i in range(feature_defn.GetFieldCount()):
                field_defn = feature_defn.GetFieldDefn(i)
                field_name = field_defn.GetNameRef()
                properties[field_name] = feature.GetField(i)

            # 组装单个 feature
            geojson_feature = {
                "type": "Feature",
                "geometry": geom_json,
                "properties": properties
            }
            geojson["features"].append(geojson_feature)

        return geojson

    def _unzip_and_get_shp(self, zip_path):
        """
        解压zip文件，返回解压后的shp文件路径
        """
        # 创建临时目录
        temp_dir = str(self.path / 'temp')
        # 解压zip文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        # 查找shp文件
        shp_path = None
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.shp'):
                    shp_path = os.path.join(root, file)
                    break
            if shp_path:
                break
        if not shp_path:
            raise FileNotFoundError('未在zip包中找到shp文件')
        return shp_path
    
    def _transform_geojson_coordinates(self, geojson_data: dict, source_epsg: str, target_epsg: str) -> dict:
        """
        Transform GeoJSON coordinates from source EPSG to target EPSG
        
        Args:
            geojson_data: GeoJSON FeatureCollection with coordinates in source CRS
            source_epsg: Source EPSG code (e.g., "4326")
            target_epsg: Target EPSG code (e.g., "3857")
            
        Returns:
            GeoJSON FeatureCollection with transformed coordinates
        """
        try:
            # Create spatial reference systems
            source_srs = osr.SpatialReference()
            source_srs.ImportFromEPSG(int(source_epsg))
            # Set axis mapping to traditional GIS order (longitude, latitude)
            source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
            target_srs = osr.SpatialReference()
            target_srs.ImportFromEPSG(int(target_epsg))
            # Set axis mapping to traditional GIS order (longitude, latitude)
            target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            
            # Transform coordinates for each feature in the FeatureCollection
            features = geojson_data.get('features', [])
            for feature in features:
                if feature.get('geometry'):
                    self._transform_geometry_ogr(feature['geometry'], source_srs, target_srs)
            
            return geojson_data
            
        except Exception as e:
            logger.error(f'Failed to transform coordinates from EPSG:{source_epsg} to EPSG:{target_epsg}: {str(e)}')
            # Return original data if transformation fails
            return geojson_data
    
    def _transform_geometry_ogr(self, geometry: dict, source_srs, target_srs) -> None:
        """
        Transform geometry using OGR's built-in transformation
        
        Args:
            geometry: GeoJSON geometry object
            source_srs: Source spatial reference system
            target_srs: Target spatial reference system
        """
        try:
            # Convert GeoJSON geometry to OGR geometry
            geom_str = json.dumps(geometry)
            ogr_geom = ogr.CreateGeometryFromJson(geom_str)
            
            if ogr_geom is None:
                logger.warning(f"Failed to create OGR geometry from: {geometry}")
                return
            
            # Assign source coordinate system
            ogr_geom.AssignSpatialReference(source_srs)
            
            # Transform to target coordinate system
            ogr_geom.TransformTo(target_srs)
            
            # Convert back to GeoJSON and update the original geometry
            transformed_json = json.loads(ogr_geom.ExportToJson())
            geometry.update(transformed_json)
            
        except Exception as e:
            logger.error(f'Failed to transform geometry using OGR: {str(e)}')
            # Keep original geometry if transformation fails
    
    def _shp_to_geojson_with_epsg(self, shp_path):
        """
        Convert shapefile to GeoJSON and detect EPSG
        
        Returns:
            tuple: (geojson_dict, epsg_code)
        """
        # 如果输入是zip文件，先解压
        if shp_path.endswith('.zip'):
            shp_path = self._unzip_and_get_shp(shp_path)
        
        # 打开 shp 文件
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(shp_path, 0)  # 0 means read-only

        if dataSource is None:
            raise FileNotFoundError(f"无法打开 {shp_path}")

        layer = dataSource.GetLayer()
        
        # 检测 EPSG 代码
        epsg_code = None
        srs = layer.GetSpatialRef()
        if srs:
            try:
                epsg_code = srs.GetAuthorityCode(None)
                if epsg_code:
                    epsg_code = str(epsg_code)
                    logger.info(f"Detected EPSG code from shapefile: {epsg_code}")
            except Exception as e:
                logger.warning(f"Failed to get EPSG code from shapefile: {str(e)}")

        # 构造 GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        # 读取.prj文件（坐标参考）
        prj_path = os.path.splitext(shp_path)[0] + '.prj'
        crs_wkt = None
        if os.path.exists(prj_path):
            with open(prj_path, 'r', encoding='utf-8') as prj_file:
                crs_wkt = prj_file.read().strip()
            
            # 如果从 SRS 获取不到 EPSG，尝试从 WKT 解析
            if not epsg_code and crs_wkt:
                try:
                    temp_srs = osr.SpatialReference()
                    temp_srs.ImportFromWkt(crs_wkt)
                    # Set axis mapping to traditional GIS order (longitude, latitude)
                    temp_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    epsg_code = temp_srs.GetAuthorityCode(None)
                    if epsg_code:
                        epsg_code = str(epsg_code)
                        logger.info(f"Detected EPSG code from PRJ file: {epsg_code}")
                except Exception as e:
                    logger.warning(f"Failed to parse EPSG from PRJ file: {str(e)}")

        if crs_wkt:
            geojson["crs_wkt"] = crs_wkt

        # 遍历所有要素
        for feature in layer:
            geom = feature.GetGeometryRef()
            # 转成 GeoJSON geometry 字典
            geom_json = json.loads(geom.ExportToJson())

            # 读取属性表
            properties = {}
            feature_defn = layer.GetLayerDefn()
            for i in range(feature_defn.GetFieldCount()):
                field_defn = feature_defn.GetFieldDefn(i)
                field_name = field_defn.GetNameRef()
                properties[field_name] = feature.GetField(i)

            # 组装单个 feature
            geojson_feature = {
                "type": "Feature",
                "geometry": geom_json,
                "properties": properties
            }
            geojson["features"].append(geojson_feature)

        return geojson, epsg_code

    def _detect_geojson_epsg(self, geojson_data: dict) -> str:
        """
        Detect EPSG code from GeoJSON data
        
        Args:
            geojson_data: GeoJSON data
            
        Returns:
            EPSG code as string, or None if not detected
        """
        try:
            # Check for CRS information in GeoJSON
            crs = geojson_data.get('crs')
            if crs:
                # Standard GeoJSON CRS format
                if crs.get('type') == 'name':
                    crs_name = crs.get('properties', {}).get('name', '')
                    if 'EPSG:' in crs_name:
                        epsg_code = crs_name.split('EPSG:')[-1]
                        logger.info(f"Detected EPSG code from GeoJSON CRS: {epsg_code}")
                        return epsg_code
                
                # Link-based CRS format
                elif crs.get('type') == 'link':
                    href = crs.get('properties', {}).get('href', '')
                    if 'epsg.io' in href or 'EPSG:' in href:
                        import re
                        match = re.search(r'EPSG:?(\d+)', href)
                        if match:
                            epsg_code = match.group(1)
                            logger.info(f"Detected EPSG code from GeoJSON link: {epsg_code}")
                            return epsg_code

            # Check for WKT CRS information (custom extension)
            crs_wkt = geojson_data.get('crs_wkt')
            if crs_wkt:
                try:
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(crs_wkt)
                    # Set axis mapping to traditional GIS order (longitude, latitude)
                    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    epsg_code = srs.GetAuthorityCode(None)
                    if epsg_code:
                        epsg_code = str(epsg_code)
                        logger.info(f"Detected EPSG code from GeoJSON WKT: {epsg_code}")
                        return epsg_code
                except Exception as e:
                    logger.warning(f"Failed to parse EPSG from WKT: {str(e)}")

            # If no CRS information found, assume WGS84 (EPSG:4326)
            logger.info("No CRS information found in GeoJSON, assuming EPSG:4326")
            return DEFAULT_EPSG

        except Exception as e:
            logger.error(f"Error detecting EPSG from GeoJSON: {str(e)}")
            # Default to WGS84 if detection fails
            return DEFAULT_EPSG

    def _ensure_feature_collection(self, geojson_data: dict) -> dict:
        """
        Ensure the GeoJSON data is in FeatureCollection format
        
        Args:
            geojson_data: GeoJSON data (could be Feature, FeatureCollection, or geometry)
            
        Returns:
            GeoJSON data in FeatureCollection format
        """
        if not isinstance(geojson_data, dict):
            raise ValueError("Invalid GeoJSON data: must be a dictionary")
        
        geojson_type = geojson_data.get('type')
        
        if geojson_type == 'FeatureCollection':
            # Already a FeatureCollection, return as is
            return geojson_data
        
        elif geojson_type == 'Feature':
            # Convert single Feature to FeatureCollection
            return {
                "type": "FeatureCollection",
                "features": [geojson_data]
            }
        
        elif geojson_type in ['Point', 'LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon', 'GeometryCollection']:
            # Convert geometry to Feature, then to FeatureCollection
            feature = {
                "type": "Feature",
                "geometry": geojson_data,
                "properties": {}
            }
            return {
                "type": "FeatureCollection", 
                "features": [feature]
            }
        
        else:
            raise ValueError(f"Unsupported GeoJSON type: {geojson_type}")
    