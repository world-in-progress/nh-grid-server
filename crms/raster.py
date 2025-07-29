import os
import io
import logging
import shutil
import c_two as cc
from pathlib import Path
from typing import Dict, Any

import rasterio
import numpy as np
from PIL import Image
from osgeo import gdal
import rasterio.windows
from rio_tiler.io import COGReader
from rio_tiler.utils import render
from rasterio.warp import transform
from rio_tiler.models import ImageData
from rasterio.features import rasterize
from shapely.geometry import shape, box
from rio_cogeo.cogeo import cog_translate
from rio_tiler.profiles import img_profiles
from rio_cogeo.profiles import cog_profiles

from icrms.iraster import IRaster, RasterOperation
from src.nh_grid_server.core.config import settings

logger = logging.getLogger(__name__)

@cc.iicrm
class Raster(IRaster):
    def __init__(self, name: str, type: str, original_tif_path: str):
        """
        初始化栅格对象
        :param original_tif_path: 原始TIF文件路径
        """
        self.name = name
        self.type = type
        self.original_tif_path = original_tif_path

        if self.type == 'dem':
            self.path = Path(f'{settings.DEM_DIR}{self.name}')
        elif self.type == 'lum':
            self.path = Path(f'{settings.LUM_DIR}{self.name}')
        self.path.mkdir(parents=True, exist_ok=True)
        
        # 资源文件夹中的原始TIF文件路径（固定名称）
        self.resource_original_tif_path = self.path / f'{self.name}.original.tif'
        # COG 文件路径（固定名称）  
        self.cog_tif_path = self.path / f'{self.name}.cog.tif'
        
        # 初始化时拷贝原始TIF到资源文件夹（如果不存在）
        self._initialize_original_tif()
        
        # 缓存栅格信息
        self._raster_info = None

        with rasterio.open(self.cog_tif_path) as src:
            self.cog_src = src
            self.cog_tif = src.read(1)

    def _initialize_original_tif(self):
        """
        初始化时拷贝原始TIF到资源文件夹，并重新计算统计信息
        """
        if not self.resource_original_tif_path.exists():
            if Path(self.original_tif_path).exists():
                # 先拷贝原始TIF文件
                shutil.copy2(self.original_tif_path, self.resource_original_tif_path)
                logger.info(f'Copied original TIF from {self.original_tif_path} to {self.resource_original_tif_path}')
                
                # 重新计算统计信息并写入文件
                self._recalculate_statistics(str(self.resource_original_tif_path))
                
            else:
                logger.error(f'Original TIF file not found: {self.original_tif_path}')
        else:
            logger.info(f'Original TIF already exists in resource folder: {self.resource_original_tif_path}')
            
            # 检查是否需要重新计算统计信息
            self._ensure_statistics_exist(str(self.resource_original_tif_path))

    def _recalculate_statistics(self, tif_path: str):
        """
        使用GDAL重新计算栅格的详细统计信息并写入文件
        """
        try:
            # 使用GDAL打开文件并计算统计信息
            dataset = gdal.Open(tif_path, gdal.GA_Update)
            if dataset is None:
                logger.error(f'Failed to open file for statistics calculation: {tif_path}')
                return
            
            band_count = dataset.RasterCount
            logger.info(f'Calculating statistics for {band_count} band(s) in {tif_path}')
            
            for band_num in range(1, band_count + 1):
                band = dataset.GetRasterBand(band_num)
                
                # 计算精确的统计信息 (approx_ok=False 表示精确计算)
                # force=True 表示强制重新计算，即使已存在统计信息
                stats = band.ComputeStatistics(False, gdal.TermProgress_nocb)
                
                if stats:
                    min_val, max_val, mean_val, std_val = stats
                    logger.info(f'Band {band_num} statistics: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}')
                    
                    # 设置统计信息到波段
                    band.SetStatistics(min_val, max_val, mean_val, std_val)
                else:
                    logger.warning(f'Failed to compute statistics for band {band_num}')
            
            # 刷新并关闭数据集，确保统计信息被写入
            dataset.FlushCache()
            dataset = None
            
            logger.info(f'Successfully recalculated and saved statistics for {tif_path}')
            
        except Exception as e:
            logger.error(f'Failed to recalculate statistics for {tif_path}: {e}')

    def _ensure_statistics_exist(self, tif_path: str):
        """
        确保TIF文件包含统计信息，如果不存在则重新计算
        """
        try:
            dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
            if dataset is None:
                return
            
            band = dataset.GetRasterBand(1)
            
            # 检查是否已有统计信息
            stats = band.GetStatistics(False, False)  # approx_ok=False, force=False
            
            if stats is None or any(stat is None for stat in stats):
                logger.info(f'Statistics missing for {tif_path}, recalculating...')
                dataset = None  # 关闭只读模式
                self._recalculate_statistics(tif_path)
            else:
                min_val, max_val, mean_val, std_val = stats
                logger.info(f'Existing statistics found: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}')
            
            if dataset:
                dataset = None
                
        except Exception as e:
            logger.warning(f'Failed to check statistics for {tif_path}: {e}')

    def _remove_auxiliary_files(self):
        """
        删除栅格文件的辅助文件，避免使用缓存的旧统计信息
        这些文件包括 .aux.xml, .ovr 等，主要是 QGIS 和其他 GIS 软件生成的缓存文件
        """
        # 需要删除的文件路径列表
        files_to_remove = [
            # 原始TIF文件的辅助文件
            str(self.resource_original_tif_path) + '.aux.xml',
            str(self.resource_original_tif_path.with_suffix('')) + '.aux.xml',
            str(self.resource_original_tif_path) + '.ovr',
            str(self.resource_original_tif_path.with_suffix('')) + '.ovr',
            
            # COG文件的辅助文件
            str(self.cog_tif_path) + '.aux.xml',
            str(self.cog_tif_path.with_suffix('')) + '.aux.xml', 
            str(self.cog_tif_path) + '.ovr',
            str(self.cog_tif_path.with_suffix('')) + '.ovr',
            
            # 其他可能的辅助文件格式
            str(self.resource_original_tif_path.with_suffix('.tif.aux.xml')),
            str(self.resource_original_tif_path.with_suffix('.tif.ovr')),
            str(self.cog_tif_path.with_suffix('.tif.aux.xml')),
            str(self.cog_tif_path.with_suffix('.tif.ovr')),
        ]
        
        removed_count = 0
        for aux_file_path in files_to_remove:
            try:
                if os.path.exists(aux_file_path):
                    os.remove(aux_file_path)
                    logger.info(f'Removed auxiliary file: {aux_file_path}')
                    removed_count += 1
            except Exception as e:
                logger.warning(f'Failed to remove auxiliary file {aux_file_path}: {e}')
        
        if removed_count > 0:
            logger.info(f'Removed {removed_count} auxiliary file(s) to ensure fresh statistics calculation')
        else:
            logger.debug('No auxiliary files found to remove')

    def _create_cloud_optimized_tif(self, input_tif_path: str = None) -> str:
        """
        创建云优化的 GeoTIFF (使用 rio-cogeo)
        :param input_tif_path: 输入的TIF文件路径，如果不提供则使用资源文件夹中的原始TIF
        :return: 云优化的 GeoTIFF 路径
        """
        # 使用传入的路径，如果没有传入则使用资源文件夹中的原始TIF
        source_tif_path = input_tif_path if input_tif_path is not None else str(self.resource_original_tif_path)
        output_tif_path = self.cog_tif_path

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
            
            # 创建云优化的 GeoTIFF（替换策略）
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
                shutil.copy2(source_tif_path, output_tif_path)
                logger.warning(f'Used original file as fallback at {output_tif_path} from {source_tif_path}')
                return str(output_tif_path)

    def get_cog_tif(self) -> str:
        """
        获取云优化的 GeoTIFF，如果不存在则创建
        :return: 云优化的 GeoTIFF 路径
        """
        # 如果COG文件存在，直接返回
        if self.cog_tif_path.exists():
            return str(self.cog_tif_path)
        
        # 创建 COG 文件
        cog_path = self._create_cloud_optimized_tif()
        return cog_path
    
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
    
    def update_by_feature(self, feature_collection: Dict[str, Any], operation: RasterOperation, value: float = 0.0) -> str:
        """
        根据 FeatureCollection 更新栅格数据
        :param feature_collection: FeatureCollection 要素集合
        :param operation: 操作类型 (RasterOperation 枚举)
        :param value: 操作值 (对于 MAX_FILL 操作此参数将被忽略)
        :return: 更新后的栅格数据路径
        """
        # 将单个FeatureCollection转换为feature_operations格式，复用update_by_features方法
        feature_operations = [{
            'feature': feature_collection,
            'operation': operation,
            'value': value
        }]
        
        return self.update_by_features(feature_operations)

    def update_by_features(self, feature_operations: list) -> str:
        """
        批量根据 FeatureCollection 更新栅格数据
        优化版本：合并所有features的边界框，只进行一次相交操作
        :param feature_operations: 包含feature、operation和value的操作列表，每个元素格式为:
                                   {'feature': feature_collection, 'operation': RasterOperation, 'value': float}
                                   其中feature必须是FeatureCollection
        :return: 更新后的栅格数据路径
        """
        cog_path = self.get_cog_tif()
        if not cog_path:
            logger.error('No COG TIF available for batch update')
            return ''
        
        if not feature_operations:
            logger.warning('No feature operations provided for batch update')
            return cog_path
        
        # 在更新前删除可能存在的辅助文件，避免使用缓存的旧统计信息
        self._remove_auxiliary_files()
        
        try:
            # 直接修改资源文件夹中的原始TIF文件
            with rasterio.open(str(self.resource_original_tif_path), 'r+') as src:
                logger.info(f'Starting optimized batch update with {len(feature_operations)} FeatureCollection operations on {self.resource_original_tif_path}')
                
                # 第一步：解析所有FeatureCollection中的features并计算合并的边界框
                all_geometries = []
                all_bounds = []
                operation_info = []  # 存储每个几何体对应的操作信息
                
                for i, feature_op in enumerate(feature_operations):
                    feature_collection = feature_op['feature']
                    operation = feature_op['operation']
                    value = feature_op.get('value', 0.0)
                    
                    # 验证是否为FeatureCollection
                    if feature_collection.get('type') != 'FeatureCollection':
                        logger.warning(f'Operation {i+1}: Expected FeatureCollection, got {feature_collection.get("type", "unknown")}, skipping')
                        continue
                    
                    # 处理FeatureCollection中的每个feature
                    features = feature_collection.get('features', [])
                    logger.debug(f'Processing FeatureCollection {i+1} with {len(features)} features')
                    
                    for j, feature in enumerate(features):
                        try:
                            geom = shape(feature['geometry'])
                            all_geometries.append(geom)
                            all_bounds.append(geom.bounds)
                            operation_info.append({'operation': operation, 'value': value})
                            logger.debug(f'FeatureCollection {i+1}, Feature {j+1}: bounds={geom.bounds}')
                        except Exception as geom_e:
                            logger.warning(f'Failed to parse geometry in FeatureCollection {i+1}, Feature {j+1}: {geom_e}')
                            continue
                
                # 计算所有features的合并边界框
                if all_bounds:
                    min_x = min(bounds[0] for bounds in all_bounds)
                    min_y = min(bounds[1] for bounds in all_bounds)
                    max_x = max(bounds[2] for bounds in all_bounds)
                    max_y = max(bounds[3] for bounds in all_bounds)
                    merged_bounds = (min_x, min_y, max_x, max_y)
                    logger.info(f'Merged bounds for all features: {merged_bounds}, total geometries: {len(all_geometries)}')
                else:
                    logger.warning('No valid geometries found')
                    return cog_path
                
                # 第二步：基于合并边界框计算栅格窗口
                left, bottom, right, top = merged_bounds
                
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
                    logger.warning('Invalid merged bounding box, no updates performed')
                    return cog_path
                
                # 计算合并窗口
                merged_window = rasterio.windows.Window(
                    col_off=ul_col, 
                    row_off=ul_row, 
                    width=lr_col - ul_col + 1, 
                    height=lr_row - ul_row + 1
                )
                
                logger.info(f'Merged window: {merged_window}')
                
                # 第三步：读取合并窗口的数据（只读取一次）
                window_data = src.read(1, window=merged_window)
                if window_data.size == 0:
                    logger.warning('No data in merged window')
                    return cog_path
                
                # 创建窗口的变换矩阵
                window_transform = rasterio.windows.transform(merged_window, src.transform)
                
                # 第四步：为每个feature创建掩膜并按顺序应用操作
                updated_data = window_data.copy()
                
                for i, (geom, op_info) in enumerate(zip(all_geometries, operation_info)):
                    operation = op_info['operation']
                    value = op_info['value']
                    
                    logger.info(f'Applying operation {i+1}/{len(all_geometries)}: {operation.value} with value {value}')
                    
                    # 检查几何是否与当前窗口相交
                    window_geom_bounds = rasterio.windows.bounds(merged_window, src.transform)
                    window_box = box(*window_geom_bounds)
                    
                    if not geom.intersects(window_box):
                        logger.debug(f'Operation {i+1}: Geometry does not intersect with window, skipping')
                        continue
                    
                    # 创建当前feature的掩膜
                    try:
                        inside_mask = rasterize(
                            [geom], 
                            out_shape=window_data.shape,
                            transform=window_transform,
                            fill=0,  # 外部像素填充为0
                            default_value=1,  # 内部像素填充为1
                            dtype=np.uint8
                        ).astype(bool)
                    except Exception as mask_e:
                        logger.warning(f'Operation {i+1}: Failed to create mask: {mask_e}, skipping')
                        continue
                    
                    # 应用操作到当前数据状态
                    if operation == RasterOperation.SET:
                        mask_valid = inside_mask
                        if src.nodata is not None:
                            mask_valid = mask_valid & (updated_data != src.nodata)
                        updated_data = np.where(mask_valid, value, updated_data)
                        
                    elif operation == RasterOperation.ADD:
                        mask_valid = inside_mask
                        if src.nodata is not None:
                            mask_valid = mask_valid & (updated_data != src.nodata)
                        updated_data = np.where(mask_valid, updated_data + value, updated_data)
                        
                    elif operation == RasterOperation.SUBTRACT:
                        mask_valid = inside_mask
                        if src.nodata is not None:
                            mask_valid = mask_valid & (updated_data != src.nodata)
                        updated_data = np.where(mask_valid, updated_data - value, updated_data)
                        
                    elif operation == RasterOperation.MAX_FILL:
                        mask_valid = inside_mask
                        if src.nodata is not None:
                            mask_valid = mask_valid & (updated_data != src.nodata)
                        
                        if np.any(mask_valid):
                            # 使用当前数据状态计算最大值
                            max_value = np.max(updated_data[mask_valid])
                            updated_data = np.where(mask_valid, max_value, updated_data)
                            logger.info(f'Operation {i+1}: Set all pixels in feature to max value: {max_value}')
                        else:
                            logger.warning(f'Operation {i+1}: No valid pixels found in feature area for max_fill operation')
                            
                    else:
                        logger.error(f'Operation {i+1}: Unsupported operation: {operation}')
                        continue
                    
                    logger.debug(f'Operation {i+1}: Applied {operation.value} operation successfully')
                
                # 第五步：一次性写入所有更新（只写入一次）
                src.write(updated_data, 1, window=merged_window)
                logger.info(f'Completed optimized batch update: applied {len(all_geometries)} geometry operations to merged window {merged_window}')
                        
            # 更新完数据后，重新计算原始TIF文件的统计信息
            logger.info('Recalculating statistics for updated original TIF file...')
            self._recalculate_statistics(str(self.resource_original_tif_path))
                        
            # 在所有操作完成后，重新生成云优化的 GeoTIFF（替换策略）
            new_cog_path = self._create_cloud_optimized_tif()
            logger.info(f'Regenerated Cloud Optimized GeoTIFF at {new_cog_path} after optimized batch update')

            # 清理全局范围缓存，因为数据已经更新
            if hasattr(self, '_global_min_max_cache'):
                delattr(self, '_global_min_max_cache')
                logger.info('Cleared global min/max cache after optimized batch raster update')
            
            return new_cog_path
                        
        except Exception as e:
            logger.error(f'Failed to perform optimized batch update raster by features: {e}')
            return str(self.cog_tif_path) if self.cog_tif_path.exists() else ""

    def sampling(self, x: float, y: float, src_crs: str = "EPSG:4326") -> float:
        """
        获取指定坐标处的栅格采样值
        :param x: X坐标
        :param y: Y坐标
        :return: 栅格值，如果坐标处无数据则返回None
        """
        cog_path = self.get_cog_tif()
        if not cog_path:
            return None

        try:
            # 处理坐标系转换
            target_x, target_y = x, y
            src = self.cog_src

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
                value = self.cog_tif[row, col]
                return float(value) if value != src.nodata else None
            else:
                logger.debug(f'Coordinates ({target_x}, {target_y}) are outside raster bounds')
                return None
        except Exception as e:
            logger.error(f'Failed to get raster value at point ({x}, {y}): {e}')
            return None

    def get_tile_png(self, x: int, y: int, z: int, encoding: str = "terrainrgb") -> bytes:
        """
        获取指定瓦片的PNG图像数据，支持多种编码格式
        
        使用rio-tiler的COGReader.tile()方法提供最高效的瓦片生成：
        1. 直接处理瓦片坐标，无需手动边界框转换
        2. 自动处理COG内部瓦片结构和金字塔
        3. 智能overview选择和数据读取
        4. 内置坐标系转换和重采样
        5. 优化的内存管理
        
        支持的编码格式：
        - "terrainrgb": Mapbox Terrain RGB编码格式，用于地形渲染和高程计算
        - "uint8": 灰度PNG格式，直接使用原栅格值作为灰度值（栅格值会被限制在0-255范围内）
        
        :param x: 瓦片X坐标
        :param y: 瓦片Y坐标  
        :param z: 缩放级别
        :param encoding: 编码格式，可选 "terrainrgb" 或 "uint8"，默认为 "terrainrgb"
        :return: 指定编码格式的256x256像素PNG图像字节数据
        """
        # 验证编码参数
        if encoding not in ["terrainrgb", "uint8"]:
            logger.warning(f'Unsupported encoding format: {encoding}, falling back to terrainrgb')
            encoding = "terrainrgb"
            
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
                
                # 根据编码格式生成PNG
                if encoding == "terrainrgb":
                    return self._create_png_from_cog_data(image_data)
                elif encoding == "uint8":
                    return self._create_uint8_png_from_cog_data(image_data)
                
        except Exception as e:
            logger.debug(f'Tile {x}/{y}/{z} not available, returning transparent tile: {e}')
            return self._create_transparent_tile(256)

    def _encode_terrain_rgb(self, dem: np.ndarray, mask: np.ndarray = None, scale_factor: float = 1.0) -> np.ndarray:
        """
        将DEM数据编码为Terrain RGB格式
        参考Mapbox Terrain RGB格式: https://docs.mapbox.com/data/tilesets/reference/mapbox-terrain-rgb-v1/
        
        Args:
            dem: DEM数据
            mask: 掩膜数据 (255表示有效数据，0表示无效数据)
            scale_factor: 高程缩放因子，1.0表示不缩放，0.1表示缩小10倍
        
        Returns:
            RGB编码的地形数据 (H, W, 3)
        """
        # 先保存原始的有效数据位置
        if mask is not None:
            valid_mask = mask == 255
            # 将掩膜外的数据设为 NaN，保持掩膜内的原始高程不变
            height = dem.copy().astype(np.float32)
            height[~valid_mask] = np.nan
        else:
            height = dem.astype(np.float32)
        
        # 只对真正的 NaN 和无穷值进行处理（不包括我们刚才设置的掩膜外NaN）
        original_nan_mask = np.isnan(dem) | np.isinf(dem)
        height = np.nan_to_num(height, nan=0, posinf=0, neginf=0)

        # 应用缩放因子
        height = height * scale_factor

        # 避免溢出：先限制 height 范围
        height = np.clip(height, -10000, 8848)  # 限制在合理的地球高程范围内

        # Mapbox Terrain RGB 编码公式
        # height_encoded = (height + 10000) * 10
        base = (height + 10000) * 10
        R = np.floor(base / (256 * 256))
        G = np.floor((base - R * 256 * 256) / 256)
        B = np.floor(base - R * 256 * 256 - G * 256)

        logger.debug(f"Height min: {np.min(height[mask == 255] if mask is not None else height)}, "
                    f"max: {np.max(height[mask == 255] if mask is not None else height)}, "
                    f"scale_factor: {scale_factor}")

        rgb = np.stack([R, G, B], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0, posinf=0, neginf=0)

        # Clamp 到 uint8 合法范围后再转换
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return rgb

    def _create_uint8_png_from_cog_data(self, image_data: ImageData) -> bytes:
        """
        从COGReader返回的ImageData创建uint8灰度格式的PNG瓦片
        
        :param image_data: COGReader返回的ImageData对象
        :return: uint8灰度格式的PNG图像字节数据
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
            
            # 处理mask，如果没有mask则创建全255的mask
            if mask is not None:
                # 确保mask是uint8格式
                final_mask = mask.astype(np.uint8)
            else:
                final_mask = np.full((height, width), 255, dtype=np.uint8)
            
            # 检查是否有有效数据
            if np.sum(final_mask == 255) == 0:
                # 无有效数据，返回透明瓦片
                logger.debug('No valid data in tile, returning transparent tile')
                return self._create_transparent_tile(256)
            
            # 将栅格值转换为uint8灰度值
            # 直接将栅格值乘以10增强对比度
            gray_data = data.astype(np.float32)
            
            # 乘以10增强对比度
            gray_data = gray_data * 10
            
            # 将栅格值限制在uint8范围内（0-255）
            gray_data = np.clip(gray_data, 0, 255)
            
            # 转换为uint8
            gray_data = gray_data.astype(np.uint8)
            
            # 应用mask：无效区域设置为透明
            rgba_data = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_data[:, :, 0] = gray_data  # R
            rgba_data[:, :, 1] = gray_data  # G  
            rgba_data[:, :, 2] = gray_data  # B
            rgba_data[:, :, 3] = final_mask  # Alpha
            
            # 创建PIL图像
            image = Image.fromarray(rgba_data, 'RGBA')
            
            # 转换为PNG字节数据
            png_buffer = io.BytesIO()
            image.save(png_buffer, format='PNG', optimize=True)
            content = png_buffer.getvalue()
            png_buffer.close()
            
            logger.debug(f'Generated uint8 grayscale PNG: size={len(content)} bytes, value_range=0-{np.max(gray_data[final_mask == 255])}')
            return content
            
        except Exception as e:
            logger.error(f'Failed to create uint8 grayscale PNG from COGReader data: {e}')
            return self._create_transparent_tile(256)

    def _create_png_from_cog_data(self, image_data: ImageData) -> bytes:
        """
        从COGReader返回的ImageData创建Terrain RGB格式的PNG瓦片
        
        :param image_data: COGReader返回的ImageData对象
        :return: Terrain RGB格式的PNG图像字节数据
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
            
            # 处理mask，如果没有mask则创建全255的mask
            if mask is not None:
                # 确保mask是uint8格式
                final_mask = mask.astype(np.uint8)
            else:
                final_mask = np.full((height, width), 255, dtype=np.uint8)
            
            # 检查是否有有效数据
            if np.sum(final_mask == 255) == 0:
                # 无有效数据，返回透明瓦片
                logger.debug('No valid data in tile, returning transparent tile')
                return self._create_transparent_tile(256)
            
            # 使用Terrain RGB编码
            # 对于DEM数据，scale_factor通常设为1.0，保持原始高程精度
            rgb = self._encode_terrain_rgb(data, final_mask, scale_factor=1.0)
            
            # 使用rio-tiler的render函数生成PNG
            # rgb shape: (H, W, 3) -> render需要 (3, H, W)
            content = render(rgb.transpose(2, 0, 1), img_format="png", **img_profiles.get("png"))
            
            logger.debug(f'Generated Terrain RGB PNG: size={len(content)} bytes')
            return content
            
        except Exception as e:
            logger.error(f'Failed to create Terrain RGB PNG from COGReader data: {e}')
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
            src = self.cog_src
            # 过滤 nodata 值
            if src.nodata is not None:
                valid_data = self.cog_tif[self.cog_tif != src.nodata]
            else:
                valid_data = self.cog_tif.flatten()

            if valid_data.size == 0:
                return {}
            
            return {
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'count': int(valid_data.size),
                'nodata_count': int(self.cog_tif.size - valid_data.size)
            }
        except Exception as e:
            logger.error(f'Failed to get raster statistics: {e}')
            return {}

    def delete_raster(self) -> Dict[str, Any]:
        """
        删除栅格资源，包括原始TIF和COG TIF文件
        """
        success = True
        messages = []
        
        # 删除原始TIF文件
        if os.path.exists(self.path):
            try:
                shutil.rmtree(self.path)
                messages.append(f'Raster deleted successfully')
            except Exception as e:
                success = False
                messages.append(f'Failed to delete original TIF: {e}')
        
        # 删除COG TIF文件
        if self.cog_tif_path.exists():
            try:
                os.remove(self.cog_tif_path)
                messages.append(f'Deleted COG TIF: {self.cog_tif_path}')
            except Exception as e:
                success = False
                messages.append(f'Failed to delete COG TIF: {e}')
        
        # 删除整个目录（如果为空）
        if self.path.exists():
            try:
                # 检查目录是否为空
                if not any(self.path.iterdir()):
                    self.path.rmdir()
                    messages.append(f'Deleted empty directory: {self.path}')
                else:
                    messages.append(f'Directory not empty, kept: {self.path}')
            except Exception as e:
                success = False
                messages.append(f'Failed to delete directory: {e}')
        
        return {
            'success': success,
            'message': '; '.join(messages) if success else f'Failed to delete raster: {"; ".join(messages)}',
        }

    def terminate(self) -> None:
        """
        清理资源
        """
        # 清理缓存
        self._raster_info = None
        if hasattr(self, '_global_min_max_cache'):
            delattr(self, '_global_min_max_cache')
        logger.info(f'Raster {self.name} terminated')