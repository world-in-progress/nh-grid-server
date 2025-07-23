import os
import io
import logging
import c_two as cc
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import rasterio
import numpy as np
from PIL import Image
from osgeo import gdal
import rasterio.windows
import geopandas as gpd
from shapely.geometry import shape
from rio_tiler.io import COGReader
from rio_tiler.models import ImageData
from rio_tiler.utils import render
from rasterio.features import rasterize
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from rasterio.warp import transform

from icrms.iraster import IRaster, RasterOperation
from src.nh_grid_server.core.config import settings

logger = logging.getLogger(__name__)

@cc.iicrm
class Raster(IRaster):
    def __init__(self, name: str, original_tif_path: str):
        """
        初始化栅格对象
        :param original_tif_path: 原始TIF文件路径
        """
        self.name = name
        self.original_tif_path = original_tif_path

        self.path = Path(f'{settings.DEM_DIR}{self.name}')
        self.path.mkdir(parents=True, exist_ok=True)
        
        # COG 文件路径（延迟创建）
        self.cog_tif_path = None
        
        # 缓存栅格信息
        self._raster_info = None

    def _create_cloud_optimized_tif(self, input_tif_path: str = None) -> str:
        """
        创建云优化的 GeoTIFF (使用 rio-cogeo)
        :param input_tif_path: 输入的TIF文件路径
        :return: 云优化的 GeoTIFF 路径
        """
        # 使用传入的路径，如果没有传入则使用默认路径
        source_tif_path = input_tif_path if input_tif_path is not None else self.original_tif_path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_tif_path = self.path / f'{self.name}.{timestamp}.cog.tif'

        if output_tif_path.exists():
            logger.info(f'Cloud Optimized GeoTIFF already exists at {output_tif_path}')
            return str(output_tif_path)

        try:
            # 使用 rio-cogeo 创建云优化的 GeoTIFF
            config = {
                "GDAL_NUM_THREADS": "ALL_CPUS",
                "GDAL_TIFF_INTERNAL_MASK": True,
                "GDAL_TIFF_OVR_BLOCKSIZE": 512,
                "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"
            }
            
            # 使用预定义的 COG 配置 (LZW 压缩)
            profile = cog_profiles.get("lzw")
            
            # 创建云优化的 GeoTIFF
            cog_translate(
                source_tif_path,
                str(output_tif_path),
                profile,
                config=config,
                in_memory=False,
                quiet=False
            )

            logger.info(f'Created Cloud Optimized GeoTIFF at {output_tif_path} from {source_tif_path}')
            return str(output_tif_path)

        except Exception as e:
            logger.error(f'Failed to create Cloud Optimized GeoTIFF with rio-cogeo: {e}')
            try:
                translate_options = gdal.TranslateOptions(
                    format='GTiff',
                    creationOptions=[
                        'COMPRESS=LZW',
                        'TILED=YES',
                        'BLOCKXSIZE=512',
                        'BLOCKYSIZE=512',
                    ]
                )
                gdal.Translate(str(output_tif_path), source_tif_path, options=translate_options)

                # Add overviews
                ds = gdal.Open(str(output_tif_path), gdal.GA_Update)
                if ds:
                    ds.BuildOverviews('NEAREST', [2, 4, 8, 16, 32])
                    ds = None

                logger.info(f'Created COG using GDAL fallback at {output_tif_path} from {source_tif_path}')
                return str(output_tif_path)
            except Exception as gdal_e:
                logger.error(f'GDAL fallback also failed: {gdal_e}')
                # Final fallback: copy original file
                import shutil
                shutil.copy2(source_tif_path, output_tif_path)
                logger.warning(f'Used original file as fallback at {output_tif_path} from {source_tif_path}')
                return str(output_tif_path)

    def get_cog_tif(self) -> str:
        """
        获取云优化的 GeoTIFF，如果不存在则创建
        :return: 云优化的 GeoTIFF 路径
        """
        # 如果已经有缓存的 COG 路径且文件存在，直接返回
        if self.cog_tif_path and Path(self.cog_tif_path).exists():
            return self.cog_tif_path

        # 检查文件夹中是否存在任何 .tif 文件
        tif_files = list(self.path.glob("*.tif"))
        if tif_files:
            # 使用第一个找到的 .tif 文件
            existing_tif = tif_files[0]
            self.cog_tif_path = str(existing_tif)
            logger.info(f'Found existing TIF file at {existing_tif}')
            return self.cog_tif_path
        
        # 创建 COG 文件
        self.cog_tif_path = self._create_cloud_optimized_tif()
        return self.cog_tif_path
    
    def _get_raster_info(self) -> Dict[str, Any]:
        """
        获取栅格信息
        """
        if self._raster_info is not None:
            return self._raster_info
            
        cog_path = self.get_cog_tif()
        if not cog_path:
            return {}
            
        try:
            with rasterio.open(cog_path) as src:
                self._raster_info = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'bounds': src.bounds,
                    'dtype': src.dtypes[0],
                    'nodata': src.nodata
                }
                return self._raster_info
        except Exception as e:
            logger.error(f'Failed to read raster info: {e}')
            return {}
    
    def update_by_feature(self, feature: Dict[str, Any], operation: RasterOperation, value: float = 0.0) -> str:
        """
        根据 GeoJSON feature 更新栅格数据
        :param feature: GeoJSON feature 要素
        :param operation: 操作类型 (RasterOperation 枚举)
        :param value: 操作值 (对于 MAX_FILL 操作此参数将被忽略)
        :return: 更新后的栅格数据路径
        """
        cog_path = self.get_cog_tif()
        if not cog_path:
            logger.error('No COG TIF available for update')
            return ''
        
        try:
            # 解析 feature 几何
            geom = shape(feature['geometry'])
            
            # 创建 GeoDataFrame
            gdf = gpd.GeoDataFrame([1], geometry=[geom])
            
            # 直接修改现有的 COG TIF 文件
            # 添加 IGNORE_COG_LAYOUT_BREAK 选项来允许修改 COG 文件
            with rasterio.open(cog_path, 'r+', IGNORE_COG_LAYOUT_BREAK='YES') as src:
                # 检查 CRS 是否匹配
                if gdf.crs is None:
                    gdf = gdf.set_crs(src.crs)
                elif gdf.crs != src.crs:
                    gdf = gdf.to_crs(src.crs)
                
                # 获取几何形状的边界框
                bounds = geom.bounds
                
                # 计算在栅格中的像素范围
                left, bottom, right, top = bounds
                
                # 转换为像素坐标
                ul_row, ul_col = rasterio.transform.rowcol(src.transform, left, top)
                lr_row, lr_col = rasterio.transform.rowcol(src.transform, right, bottom)
                
                # 确保坐标在栅格范围内
                ul_row = max(0, min(ul_row, src.height - 1))
                ul_col = max(0, min(ul_col, src.width - 1))
                lr_row = max(0, min(lr_row, src.height - 1))
                lr_col = max(0, min(lr_col, src.width - 1))
                
                # 确保边界框有效
                if ul_row >= lr_row or ul_col >= lr_col:
                    logger.warning('Invalid bounding box for feature')
                    return cog_path
                
                # 读取边界框区域的数据
                window = rasterio.windows.Window(
                    col_off=ul_col, 
                    row_off=ul_row, 
                    width=lr_col - ul_col + 1, 
                    height=lr_row - ul_row + 1
                )
                
                # 读取窗口数据
                window_data = src.read(1, window=window)
                if window_data.size == 0:
                    logger.warning('No data in window')
                    return cog_path
                
                # 创建窗口的变换矩阵
                window_transform = rasterio.windows.transform(window, src.transform)
                
                # 使用 rasterize 创建几何掩膜
                inside_mask = rasterize(
                    [geom], 
                    out_shape=window_data.shape,
                    transform=window_transform,
                    fill=0,  # 外部像素填充为0
                    default_value=1,  # 内部像素填充为1
                    dtype=np.uint8
                ).astype(bool)
                
                # 创建更新后的数据副本
                updated_data = window_data.copy()
                
                # 只对几何形状内部的像素进行操作
                if operation == RasterOperation.SET:
                    # 直接赋值，但只对几何形状内部且非nodata的像素
                    mask_valid = inside_mask  # 几何形状内部
                    if src.nodata is not None:
                        mask_valid = mask_valid & (window_data != src.nodata)  # 排除原始nodata
                    updated_data = np.where(mask_valid, value, window_data)
                elif operation == RasterOperation.ADD:
                    # 加值操作，只对几何形状内部且非nodata的像素
                    mask_valid = inside_mask  # 几何形状内部
                    if src.nodata is not None:
                        mask_valid = mask_valid & (window_data != src.nodata)  # 排除原始nodata
                    updated_data = np.where(mask_valid, window_data + value, window_data)
                elif operation == RasterOperation.SUBTRACT:
                    # 减值操作，只对几何形状内部且非nodata的像素
                    mask_valid = inside_mask  # 几何形状内部
                    if src.nodata is not None:
                        mask_valid = mask_valid & (window_data != src.nodata)  # 排除原始nodata
                    updated_data = np.where(mask_valid, window_data - value, window_data)
                elif operation == RasterOperation.MAX_FILL:
                    # 将feature范围内所有像素设置为该范围内的最高值
                    mask_valid = inside_mask  # 几何形状内部
                    if src.nodata is not None:
                        mask_valid = mask_valid & (window_data != src.nodata)  # 排除原始nodata
                    
                    if np.any(mask_valid):
                        # 获取几何形状内部有效像素的最大值
                        max_value = np.max(window_data[mask_valid])
                        updated_data = np.where(mask_valid, max_value, window_data)
                        logger.info(f'Set all pixels in feature to max value: {max_value}')
                    else:
                        logger.warning('No valid pixels found in feature area for max_fill operation')
                        updated_data = window_data
                else:
                    logger.error(f'Unsupported operation: {operation}')
                    return cog_path
                
                # 直接写入更新的数据到原始COG文件
                src.write(updated_data, 1, window=window)
                
                logger.info(f'Updated original COG TIF at {cog_path}')
                        
            # 直接使用修改后的 COG 文件重新生成云优化的 GeoTIFF
            new_cog_path = self._create_cloud_optimized_tif(cog_path)
            self.cog_tif_path = new_cog_path

            os.remove(cog_path)  # 删除原始 COG 文件
            logger.info(f'Removed original COG TIF at {cog_path}')

            # 清理全局范围缓存，因为数据已经更新
            if hasattr(self, '_global_min_max_cache'):
                delattr(self, '_global_min_max_cache')
                logger.info('Cleared global min/max cache after raster update')

            logger.info(f'Regenerated Cloud Optimized GeoTIFF at {new_cog_path}')
            
            return new_cog_path
                        
        except Exception as e:
            logger.error(f'Failed to update raster by feature: {e}')
            return cog_path

    def sampling(self, x: float, y: float) -> float:
        """
        获取指定坐标处的栅格采样值
        :param x: X坐标
        :param y: Y坐标
        :return: 栅格值，如果坐标处无数据则返回None
        """
        cog_path = self.get_cog_tif()
        if not cog_path:
            return None
            
        src_crs = "EPSG:4326"

        try:
            with rasterio.open(cog_path) as src:
                # 处理坐标系转换
                target_x, target_y = x, y
                
                # 如果指定了源坐标系且与栅格坐标系不同，进行坐标转换
                if src_crs is not None and src.crs is not None:
                    if src_crs != src.crs.to_string():
                        try:
                            # 使用 rasterio.warp.transform 进行坐标转换
                            target_x, target_y = transform(src_crs, src.crs, [x], [y])
                            target_x, target_y = target_x[0], target_y[0]
                            logger.debug(f'Transformed coordinates from {src_crs} ({x}, {y}) to {src.crs} ({target_x}, {target_y})')
                        except Exception as transform_e:
                            logger.error(f'Failed to transform coordinates from {src_crs} to {src.crs}: {transform_e}')
                            return None
                
                # 将地理坐标转换为像素坐标
                row, col = rasterio.transform.rowcol(src.transform, target_x, target_y)
                
                # 检查坐标是否在栅格范围内
                if 0 <= row < src.height and 0 <= col < src.width:
                    value = src.read(1)[row, col]
                    return float(value) if value != src.nodata else None
                else:
                    logger.debug(f'Coordinates ({target_x}, {target_y}) are outside raster bounds')
                    return None
        except Exception as e:
            logger.error(f'Failed to get raster value at point ({x}, {y}): {e}')
            return None

    def get_tile_png(self, x: int, y: int, z: int) -> bytes:
        """
        获取指定瓦片的PNG图像数据，使用COGReader优化
        
        使用rio-tiler的COGReader.tile()方法提供最高效的瓦片生成：
        1. 直接处理瓦片坐标，无需手动边界框转换
        2. 自动处理COG内部瓦片结构和金字塔
        3. 智能overview选择和数据读取
        4. 内置坐标系转换和重采样
        5. 优化的内存管理
        
        :param x: 瓦片X坐标
        :param y: 瓦片Y坐标  
        :param z: 缩放级别
        :return: 256x256像素的PNG图像字节数据
        """
        cog_path = self.get_cog_tif()
        if not cog_path:
            logger.debug(f'No COG file available for tile generation {x}/{y}/{z}')
            return self._create_transparent_tile(256)
            
        try:
            # 使用COGReader直接获取瓦片
            with COGReader(cog_path) as cog:
                image_data: ImageData = cog.tile(x, y, z, tilesize=256)
                
                # 检查是否有有效数据
                if image_data.data.size == 0:
                    logger.debug(f'No data available for tile {x}/{y}/{z}')
                    return self._create_transparent_tile(256)
                
                # 使用COGReader返回的ImageData生成PNG
                return self._create_png_from_cog_data(image_data)
                
        except Exception as e:
            logger.debug(f'Tile {x}/{y}/{z} not available, returning transparent tile: {e}')
            return self._create_transparent_tile(256)

    def _create_png_from_cog_data(self, image_data: ImageData) -> bytes:
        """
        从COGReader返回的ImageData创建PNG瓦片
        
        :param image_data: COGReader返回的ImageData对象
        :return: PNG图像字节数据
        """
        try:
            # 获取数据数组，通常是 (bands, height, width) 格式
            data = image_data.data
            mask = image_data.mask if image_data.mask is not None else None
            
            # 如果是多波段，只使用第一个波段
            if len(data.shape) == 3 and data.shape[0] > 1:
                data = data[0]
                if mask is not None and len(mask.shape) == 3:
                    mask = mask[0]
            elif len(data.shape) == 3:
                data = data[0]
                if mask is not None and len(mask.shape) == 3:
                    mask = mask[0]
            
            # 确保数据是2D数组
            if len(data.shape) != 2:
                logger.error(f'Unexpected data shape: {data.shape}')
                return self._create_transparent_tile(256)
            
            height, width = data.shape
            
            # 处理mask，如果没有mask则认为所有数据都有效
            if mask is not None:
                valid_mask = mask.astype(bool)
            else:
                valid_mask = np.ones_like(data, dtype=bool)
            
            # 获取有效数据进行标准化
            valid_data = data[valid_mask]
            
            if valid_data.size == 0:
                # 无有效数据，返回透明瓦片
                return self._create_transparent_tile(256)
            
            # 使用全局的最小值和最大值进行标准化，而不是瓦片局部的值
            # 这样可以确保所有瓦片使用相同的色彩映射，避免边界问题
            global_min, global_max = self._get_global_min_max()
            
            if global_max > global_min:
                # 使用全局范围进行标准化
                normalized = np.clip((data - global_min) / (global_max - global_min) * 255, 0, 255)
            else:
                # 所有值相同，使用中等灰度
                normalized = np.full_like(data, 128, dtype=np.float32)
            
            # 创建RGBA数组
            rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
            
            # 设置灰度值
            gray_values = np.clip(normalized, 0, 255).astype(np.uint8)
            
            # 设置RGB通道（灰度渲染）
            rgba_array[:, :, 0] = gray_values  # Red
            rgba_array[:, :, 1] = gray_values  # Green
            rgba_array[:, :, 2] = gray_values  # Blue
            
            # 设置透明度通道
            alpha_values = np.where(valid_mask, 255, 0).astype(np.uint8)
            rgba_array[:, :, 3] = alpha_values
            
            # 生成Web优化的PNG
            image = Image.fromarray(rgba_array, 'RGBA')
            
            # PNG压缩优化
            png_buffer = io.BytesIO()
            image.save(
                png_buffer, 
                format='PNG',
                optimize=True,
                compress_level=6,  # 平衡压缩率和速度
                pnginfo=None  # 不包含元数据以减小文件大小
            )
            
            png_data = png_buffer.getvalue()
            png_buffer.close()
            
            logger.debug(f'Generated COGReader PNG: size={len(png_data)} bytes')
            return png_data
            
        except Exception as e:
            logger.error(f'Failed to create PNG from COGReader data: {e}')
            return self._create_transparent_tile(256)

    def _get_global_min_max(self):
        """
        获取整个栅格数据的全局最小值和最大值，用于一致的颜色映射
        这样可以确保所有瓦片使用相同的数据范围进行标准化
        """
        if hasattr(self, '_global_min_max_cache'):
            return self._global_min_max_cache
        
        cog_path = self.get_cog_tif()
        if not cog_path:
            return 0, 255
        
        try:
            with rasterio.open(cog_path) as src:
                # 使用栅格统计信息来获取全局范围，避免读取整个文件
                stats = src.statistics(1, approx=True)
                if stats and stats.min is not None and stats.max is not None:
                    global_min = stats.min
                    global_max = stats.max
                else:
                    # 如果统计信息不可用，使用采样方法
                    # 读取多个小块来估算全局范围
                    global_min = float('inf')
                    global_max = float('-inf')
                    
                    # 采样策略：读取几个代表性的块
                    height, width = src.height, src.width
                    sample_size = min(1024, width // 4, height // 4)  # 采样块大小
                    
                    # 在不同位置采样
                    sample_positions = [
                        (0, 0),  # 左上角
                        (width - sample_size, 0),  # 右上角
                        (0, height - sample_size),  # 左下角
                        (width - sample_size, height - sample_size),  # 右下角
                        (width // 2 - sample_size // 2, height // 2 - sample_size // 2)  # 中心
                    ]
                    
                    for col_off, row_off in sample_positions:
                        if col_off >= 0 and row_off >= 0:
                            window = rasterio.windows.Window(
                                col_off=col_off, 
                                row_off=row_off, 
                                width=min(sample_size, width - col_off),
                                height=min(sample_size, height - row_off)
                            )
                            
                            sample_data = src.read(1, window=window)
                            if src.nodata is not None:
                                sample_data = sample_data[sample_data != src.nodata]
                            
                            if sample_data.size > 0:
                                global_min = min(global_min, np.min(sample_data))
                                global_max = max(global_max, np.max(sample_data))
                    
                    # 如果采样失败，使用默认值
                    if global_min == float('inf') or global_max == float('-inf'):
                        global_min, global_max = 0, 255
                
                # 缓存结果
                self._global_min_max_cache = (float(global_min), float(global_max))
                logger.info(f'Global value range for {self.name}: {global_min} - {global_max}')
                return self._global_min_max_cache
                
        except Exception as e:
            logger.error(f'Failed to get global min/max values: {e}')
            # 使用默认值
            self._global_min_max_cache = (0, 255)
            return self._global_min_max_cache

    def _create_transparent_tile(self, size: int) -> bytes:
        """创建透明瓦片作为错误处理或空白区域"""
        try:
            empty_image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            png_buffer = io.BytesIO()
            empty_image.save(png_buffer, format='PNG', optimize=True)
            png_data = png_buffer.getvalue()
            png_buffer.close()
            return png_data
        except:
            return b''
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取栅格的元数据信息
        :return: 包含bbox、epsg、最小值、最大值和无效值的字典
        """
        cog_path = self.get_cog_tif()
        if not cog_path:
            return {}
            
        try:
            with rasterio.open(cog_path) as src:
                # 获取边界框
                bounds = src.bounds
                bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
                
                # 获取EPSG代码
                epsg = None
                if src.crs and src.crs.to_epsg():
                    epsg = src.crs.to_epsg()
                elif src.crs:
                    epsg = src.crs.to_string()
                
                # 使用全局最小值和最大值，确保与瓦片渲染一致
                min_val, max_val = self._get_global_min_max()
                
                return {
                    'bbox': bbox,
                    'epsg': epsg,
                    'min_value': min_val,
                    'max_value': max_val,
                    'nodata_value': src.nodata,
                    'width': src.width,
                    'height': src.height,
                    'dtype': str(src.dtypes[0]),
                    'transform': list(src.transform)[:6],  # 转换为列表以便JSON序列化
                    'crs': src.crs.to_string() if src.crs else None
                }
                
        except Exception as e:
            logger.error(f'Failed to get raster metadata: {e}')
            return {}

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取栅格统计信息
        :return: 统计信息字典
        """
        cog_path = self.get_cog_tif()
        if not cog_path:
            return {}
            
        try:
            with rasterio.open(cog_path) as src:
                data = src.read(1)
                
                # 过滤 nodata 值
                if src.nodata is not None:
                    valid_data = data[data != src.nodata]
                else:
                    valid_data = data.flatten()
                
                if valid_data.size == 0:
                    return {}
                
                return {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'count': int(valid_data.size),
                    'nodata_count': int(data.size - valid_data.size)
                }
        except Exception as e:
            logger.error(f'Failed to get raster statistics: {e}')
            return {}

    def terminate(self) -> None:
        """
        清理资源
        """
        # 清理缓存
        self._raster_info = None
        if hasattr(self, '_global_min_max_cache'):
            delattr(self, '_global_min_max_cache')
        logger.info(f'Raster {self.name} terminated')