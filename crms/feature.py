import c_two as cc
from icrms.ifeature import IFeature
import logging
import os
import json
from pathlib import Path
from typing import Any
from osgeo import ogr
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@cc.iicrm
class Feature(IFeature):
    """
    CRM
    =
    The Feature Resource.  
    Feature is a feature file that can be uploaded to the resource pool.  
    """
    def __init__(self, feature_path: str):
        """Method to initialize Feature

        Args:
            feature_path (str): path to the feature file
        """
        self.feature_path = Path(feature_path)
        os.makedirs(feature_path, exist_ok=True)
        logger.info(f'Feature initialized with feature path: {feature_path}')

    def upload_feature(self, file_path: str, file_type: str, feature_type: str) -> dict[str, bool | str]:
        """
        Upload a feature file to the resource pool
        """

        logger.info(f'Uploading feature in crm: {file_path}-{file_type}-{feature_type}')

        if file_type == 'json':
            return {
                'success': True,
                'file_path': file_path,
            }
        elif file_type == 'shp':
            geojson = self._shp_to_geojson(file_path)
            # 将geojson写入文件
            file_path = str(self.feature_path / feature_type / os.path.basename(file_path) + '.json')
            with open(file_path, 'w') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)
            logger.info(f'Converting shp file to geojson: {file_path}')
            return {
                'success': True,
                'file_path': file_path,
            }
        else:
            return {
                'success': False,
                'file_path': '',
            }

    def save_uploaded_feature(self, file_path: str, feature_type: str, feature_json: dict[str, Any], is_edited: bool) -> dict[str, bool | str]:
        """
        Save feature to resource pool
        """
        try:
            # 如果文件被编辑，复制到资源池
            if is_edited:
                # 获取文件名
                file_name = os.path.basename(file_path)
                # 复制文件到资源池
                target_path = os.path.join(self.feature_path, feature_type, file_name + '.json')
                # 创建目录
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # 将json写入
                with open(target_path, 'w') as f:
                    json.dump(feature_json, f, ensure_ascii=False, indent=2)
                resource_path = target_path
            else:
                # 如果文件未被编辑，直接使用原路径
                resource_path = file_path
            
            # 调用资源树挂载接口
            # TODO: 实现资源树挂载逻辑
            
            return {
                'success': True,
                'message': "Feature saved successfully",
                'resource_path': resource_path
            }
        except Exception as e:
            logger.error(f'Failed to save feature: {str(e)}')
            return {
                'success': False,
                'message': str(e),
                'resource_path': ""
            }
        
    def save_feature(self, feature_name: str, feature_type: str, feature_json: dict[str, Any]) -> dict[str, bool | str]:
        try:
            # 构造目标文件路径
            target_path = os.path.join(self.feature_path, feature_type, feature_name + '.json')
            # 创建目录
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # 写入格式化的json内容
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(feature_json, f, ensure_ascii=False, indent=2)
            return {
                'success': True,
                'message': 'Feature saved successfully',
                'resource_path': target_path
            }
        except Exception as e:
            logger.error(f'Failed to save feature: {str(e)}')
            return {
                'success': False,
                'message': str(e),
                'resource_path': ''
            }
        
    def get_feature_json(self, feature_name: str, feature_type: str) -> dict[str, Any]:
        """
        Get feature json
        """
        file_path = os.path.join(self.feature_path, feature_type, feature_name + '.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

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
        temp_dir = str(self.feature_path / 'temp')
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