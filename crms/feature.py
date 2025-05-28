import c_two as cc
from icrms.ifeature import IFeature, UploadInfo, SaveInfo
import logging
import os
import json
import shutil
from typing import Any

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
    def __init__(self, resource_pool_path: str):
        """Method to initialize Feature

        Args:
            resource_pool_path (str): path to the resource pool directory
        """
        self.resource_pool_path = resource_pool_path
        os.makedirs(resource_pool_path, exist_ok=True)
        logger.info(f'Feature initialized with resource pool path: {resource_pool_path}')

    def upload_feature(self, file_path: str, file_type: str, patch_name: str) -> UploadInfo:
        """
        Upload a feature file to the resource pool
        """
        if file_type == 'json':
            return UploadInfo(
                success=True,
                file_path=file_path,
                patch_name=patch_name
            )
        elif file_type == 'shp':
            # TODO: 将shp文件转换为geojson
            return UploadInfo(
                success=True,
                file_path=file_path,
                patch_name=patch_name
            )
        else:
            return UploadInfo(
                success=False,
                file_path='',
                patch_name=''
            )

    def save_vector_feature(self, info: UploadInfo, is_edited: bool) -> SaveInfo:
        """
        Save feature to resource pool
        """
        try:
            if not info.success:
                return SaveInfo(
                    success=False,
                    message="Upload failed",
                    resource_path="",
                    is_edited=False
                )
            
            if is_edited:
                # 如果文件被编辑，复制到资源池
                file_name = os.path.basename(info.file_path)
                target_path = os.path.join(self.resource_pool_path, info.patch_name, file_name)
                shutil.copy2(info.file_path, target_path)
                resource_path = target_path
            else:
                # 如果文件未被编辑，直接使用原路径
                resource_path = info.file_path
            
            # 调用资源树挂载接口
            # TODO: 实现资源树挂载逻辑
            
            return SaveInfo(
                success=True,
                message="Feature saved successfully",
                resource_path=resource_path
            )
        except Exception as e:
            logger.error(f'Failed to save feature: {str(e)}')
            return SaveInfo(
                success=False,
                message=str(e),
                resource_path=""
            )

    def get_feature_list_of_patch(self, patch_name: str) -> list[dict[str, Any]]:
        """
        Get list of features in the resource pool
        """
        try:
            features = []
            for root, dirs, files in os.walk(self.resource_pool_path):
                for file in files:
                    if file.endswith(('.json', '.shp')):
                        file_path = os.path.join(root, file)
                        features.append({
                            'file_path': file_path,
                            'file_type': os.path.splitext(file)[1][1:],
                            'patch_name': os.path.basename(os.path.dirname(file_path))
                        })
            return features
        except Exception as e:
            logger.error(f'Failed to get feature list: {str(e)}')
            return []