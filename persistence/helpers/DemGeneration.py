from osgeo import gdal, osr
import numpy as np
import os

from .png2huv import process_terrain_data, create_dem_rgba_image, save_image
import time

class GDALTransformer:
    """使用 GDAL 实现的坐标转换器，替代 pyproj.Transformer"""
    
    def __init__(self, source_epsg, target_epsg):
        """
        初始化 GDAL 坐标转换器
        
        Args:
            source_epsg: 源坐标系 EPSG 代码
            target_epsg: 目标坐标系 EPSG 代码
        """
        self.source_epsg = source_epsg
        self.target_epsg = target_epsg
        
        # 创建源坐标系
        self.source_srs = osr.SpatialReference()
        self.source_srs.ImportFromEPSG(source_epsg)
        # 设置传统GIS轴顺序（x=经度, y=纬度）
        self.source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        # 创建目标坐标系
        self.target_srs = osr.SpatialReference()
        self.target_srs.ImportFromEPSG(target_epsg)
        # 设置传统GIS轴顺序（x=经度, y=纬度）
        self.target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        # 创建坐标转换对象
        self.coord_transform = osr.CoordinateTransformation(self.source_srs, self.target_srs)
    
    @classmethod
    def from_crs(cls, source_crs, target_crs, always_xy=True):
        """
        从 CRS 字符串创建转换器，模拟 pyproj.Transformer.from_crs
        
        Args:
            source_crs: 源坐标系字符串，格式如 "EPSG:2326"
            target_crs: 目标坐标系字符串，格式如 "EPSG:4326" 
            always_xy: 是否始终按 x,y 顺序（GDAL 默认按 x,y，此参数为兼容性）
        
        Returns:
            GDALTransformer 实例
        """
        # 从字符串中提取 EPSG 代码
        source_epsg = int(source_crs.split(':')[1])
        target_epsg = int(target_crs.split(':')[1])
        
        return cls(source_epsg, target_epsg)
    
    def transform(self, x, y):
        """
        执行坐标转换
        
        Args:
            x: x 坐标，可以是单个值或数组
            y: y 坐标，可以是单个值或数组
        
        Returns:
            tuple: (transformed_x, transformed_y)
        """
        # 处理单个点
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            point = self.coord_transform.TransformPoint(x, y)
            return point[0], point[1]
        
        # 处理数组
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        
        # 确保是1维数组
        x_flat = x_array.flatten()
        y_flat = y_array.flatten()
        
        # 批量转换
        transformed_x = []
        transformed_y = []
        
        for xi, yi in zip(x_flat, y_flat):
            point = self.coord_transform.TransformPoint(xi, yi)
            transformed_x.append(point[0])
            transformed_y.append(point[1])
        
        # 转换为 numpy 数组并恢复原始形状
        transformed_x = np.array(transformed_x).reshape(x_array.shape)
        transformed_y = np.array(transformed_y).reshape(y_array.shape)
        
        return transformed_x, transformed_y

def process_dem_to_image_from_datasets(input_dem_path: str, output_path: str, file_suffix: str = "", 
                                      transform_crs: bool = True, source_epsg: int = 2326, target_epsg: int = 3857):
    """
    从DEM文件处理数据到图像，并输出尺寸与统计信息
    """
    
    print("开始处理DEM文件...")
    start_time = time.time()
    
    try:
        # 打开DEM文件
        dataset = gdal.Open(input_dem_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"无法打开DEM文件: {input_dem_path}")
        
        # 坐标转换处理
        if transform_crs and source_epsg != target_epsg:
            print(f"执行坐标转换: EPSG:{source_epsg} -> EPSG:{target_epsg}")
            
            # 创建坐标转换器
            transformer = GDALTransformer.from_crs(f"EPSG:{source_epsg}", f"EPSG:{target_epsg}", always_xy=True)
            
            # 获取当前的地理变换参数
            geo_transform = dataset.GetGeoTransform()
            
            # 计算四个角点的坐标
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            
            corners = [
                (geo_transform[0], geo_transform[3]),  # 左上角
                (geo_transform[0] + width * geo_transform[1], geo_transform[3]),  # 右上角
                (geo_transform[0], geo_transform[3] + height * geo_transform[5]),  # 左下角
                (geo_transform[0] + width * geo_transform[1], geo_transform[3] + height * geo_transform[5])  # 右下角
            ]
            
            # 转换角点坐标
            transformed_corners = []
            for x, y in corners:
                tx, ty = transformer.transform(x, y)
                transformed_corners.append((tx, ty))
            
            # 更新地理变换参数
            min_x = min(corner[0] for corner in transformed_corners)
            max_y = max(corner[1] for corner in transformed_corners)
            
            pixel_size_x = (max(corner[0] for corner in transformed_corners) - min_x) / width
            pixel_size_y = -(max_y - min(corner[1] for corner in transformed_corners)) / height
            
            new_geo_transform = (min_x, pixel_size_x, 0, max_y, 0, pixel_size_y)
            dataset.SetGeoTransform(new_geo_transform)
            
            # 更新空间参考系统
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(target_epsg)
            dataset.SetProjection(srs.ExportToWkt())
        
        # 从数据集读取DEM数据
        dem_data = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
        
        if dem_data is None:
            raise RuntimeError("无法从数据集读取DEM数据")
        
        # 获取图片尺寸
        height, width = dem_data.shape
        image_size = {
            'width': width,
            'height': height
        }
        
        # 处理地形数据
        norm_data, min_height, max_height = process_terrain_data(dem_data)

        # 存储统计信息，使用与其他函数一致的格式
        field_stats = {}
        field_stats['dem'] = {
            'min_value': float(min_height),
            'max_value': float(max_height)
        }
        
        # 获取地理变换参数用于计算角点坐标
        geo_transform = dataset.GetGeoTransform()
        
        # 计算4个角点在当前坐标系下的坐标
        top_left_x = geo_transform[0]
        top_left_y = geo_transform[3]
        bottom_right_x = geo_transform[0] + width * geo_transform[1]
        bottom_right_y = geo_transform[3] + height * geo_transform[5]
        
        # 将角点坐标转换为4326坐标系（经纬度）
        # 获取当前坐标系的EPSG代码
        current_srs = osr.SpatialReference()
        current_srs.ImportFromWkt(dataset.GetProjection())
        
        # 尝试获取EPSG代码
        current_epsg = None
        if current_srs.GetAuthorityCode(None):
            current_epsg = int(current_srs.GetAuthorityCode(None))
        else:
            # 如果无法获取EPSG代码，根据坐标转换情况推断
            if transform_crs and source_epsg != target_epsg:
                current_epsg = target_epsg
            else:
                current_epsg = source_epsg
        
        # 使用GDALTransformer进行坐标转换
        transformer_to_4326 = GDALTransformer.from_crs(f"EPSG:{current_epsg}", "EPSG:4326", always_xy=True)
        
        # 转换角点坐标到WGS84（经纬度）
        top_left_lon, top_left_lat = transformer_to_4326.transform(top_left_x, top_left_y)
        bottom_right_lon, bottom_right_lat = transformer_to_4326.transform(bottom_right_x, bottom_right_y)
        
        # 创建DEM RGBA图像
        image = create_dem_rgba_image(norm_data)
        
        # 保存图像，使用file_suffix作为文件名后缀
        if file_suffix:
            output_dem_path = os.path.join(output_path, f'dem_{file_suffix}.png')
            output_stats_path = os.path.join(output_path, f'dem_stats_{file_suffix}.txt')
        else:
            output_dem_path = os.path.join(output_path, 'dem.png')
            output_stats_path = os.path.join(output_path, 'dem_stats.txt')
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_dem_path), exist_ok=True)
        
        save_image(image, output_dem_path)
        
        # 写入统计信息到txt文件
        try:
            with open(output_stats_path, 'w', encoding='utf-8') as stats_file:
                # 写入表头
                stats_file.write("width height terrainHeightMin terrainHeightMax topLeftLon topLeftLat bottomRightLon bottomRightLat\n")
                # 写入数据
                stats_file.write(f"{width} {height} {field_stats['dem']['min_value']} {field_stats['dem']['max_value']} "
                               f"{top_left_lon} {top_left_lat} {bottom_right_lon} {bottom_right_lat}\n")
            print(f"统计信息已写入: {output_stats_path}")
        except Exception as e:
            print(f"写入统计信息文件时出错: {e}")
        
        # 输出统计信息到日志
        
        processing_time = time.time() - start_time
        print(f"DEM文件处理完成，耗时: {processing_time:.2f} 秒")
        
        # 清理资源
        dataset = None
        
        return field_stats, image_size 
        
    except Exception as e:
        print(f"处理DEM文件时出错: {e}")
        # 确保资源被清理
        if 'dataset' in locals():
            dataset = None
        raise

 
# dem_input_path = "./HK_dem/Digital Terrain Model.tif"
# dem_output_path = "./dem/"
# dem_field_stats, dem_image_size = process_dem_to_image_from_datasets(dem_input_path, dem_output_path)
# print(f"字段统计信息: {dem_field_stats}")
# print(f"图片尺寸: {dem_image_size['width']} x {dem_image_size['height']}")