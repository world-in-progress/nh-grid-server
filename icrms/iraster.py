import c_two as cc
from typing import Dict, Any
from enum import Enum

class RasterOperation(Enum):
    """栅格操作类型枚举"""
    SET = "set"           # 直接设置值
    ADD = "add"           # 加值操作
    SUBTRACT = "subtract" # 减值操作
    MAX_FILL = "max_fill" # 将范围内所有像素设置为该范围内的最高值

@cc.icrm
class IRaster:

    def get_cog_tif(self) -> str:
        """
        获取云优化的 GeoTIFF
        :return: 云优化的 GeoTIFF
        """
        ...

    def update_by_feature(self, feature: Dict[str, Any], operation: RasterOperation, value: float = 0.0) -> str:
        """
        根据 GeoJSON feature 更新栅格数据
        :param feature: GeoJSON feature 要素
        :param operation: 操作类型 ('set', 'add', 'subtract')
        :param value: 操作值
        :return: 更新后的栅格数据路径
        """
        ...

    def update_by_features(self, feature_operations: list) -> str:
        """
        批量根据 GeoJSON feature 更新栅格数据
        :param feature_operations: 包含feature、operation和value的操作列表
        :return: 更新后的栅格数据路径
        """
        ...

    def get_tile_png(self, x: int, y: int, z: int) -> bytes:
        """
        获取指定瓦片的PNG图像数据
        :param x: 瓦片X坐标
        :param y: 瓦片Y坐标  
        :param z: 缩放级别
        :return: PNG图像的字节数据
        """
        ...

    def sampling(self, x: float, y: float) -> float:
        """
        获取指定坐标处的栅格采样值
        :param x: X坐标
        :param y: Y坐标
        :return: 栅格值，如果坐标处无数据则返回None
        """
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取栅格的元数据信息
        :return: 包含bbox、epsg、最小值、最大值和无效值的字典
        """
        ...

    def delete_raster(self) -> Dict[str, Any]:
        """
        删除栅格数据
        :return: 删除结果信息
        """
        ...