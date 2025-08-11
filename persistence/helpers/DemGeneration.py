import json
from osgeo import gdal, osr
import numpy as np
import os
import time
from .dataset2huv import process_terrain_data, create_dem_rgba_image, save_image

def get_dataset_bounds(dataset):
    """
    获取数据集四个角点在 EPSG:4326 坐标系下的经纬度坐标

    参数:
        dataset: 一个 GDAL 数据集对象

    返回:
        字典格式：
        {
            "upper_left": {"lon": ..., "lat": ...},
            "lower_left": {"lon": ..., "lat": ...},
            "lower_right": {"lon": ..., "lat": ...},
            "upper_right": {"lon": ..., "lat": ...}
        }
    """
    # 获取地理变换参数
    gt = dataset.GetGeoTransform()
    if gt is None:
        raise ValueError("无地理变换信息")

    # 数据集大小
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 四个角点的原始坐标
    ul_x, ul_y = gt[0], gt[3]  # 左上角
    lr_x, lr_y = gt[0] + width * gt[1] + height * gt[2], gt[3] + width * gt[4] + height * gt[5]  # 右下角
    ll_x, ll_y = ul_x, lr_y  # 左下角
    ur_x, ur_y = lr_x, ul_y  # 右上角

    # 返回结果
    return {
        "lower_left": {"lon": ll_x, "lat": ll_y},
        "lower_right": {"lon": lr_x, "lat": lr_y},
        "upper_right": {"lon": ur_x, "lat": ur_y},
        "upper_left": {"lon": ul_x, "lat": ul_y},
    }

def reproject_dataset(src_dataset, target_epsg=3857, resampling=gdal.GRA_Bilinear):
    """
    将栅格数据集重新投影到目标坐标系（默认 EPSG:3857），并返回元信息字典
    """
    print("开始投影转换...")

    # 读取原始坐标系
    src_proj = osr.SpatialReference()
    src_wkt = src_dataset.GetProjection()
    if not src_wkt:
        raise RuntimeError("源数据缺少投影信息，请检查输入 DEM 文件。")

    src_proj.ImportFromWkt(src_wkt)

    # 创建目标坐标系
    tgt_proj = osr.SpatialReference()
    tgt_proj.ImportFromEPSG(target_epsg)
    dst_wkt = tgt_proj.ExportToWkt()

    # 获取原始数据的 geoTransform 和尺寸信息
    geo_transform = src_dataset.GetGeoTransform()
    print(geo_transform)
    width = src_dataset.RasterXSize
    height = src_dataset.RasterYSize

    if width <= 0 or height <= 0:
        raise RuntimeError("源数据尺寸异常（宽或高 <= 0）")

    if geo_transform is None:
        raise RuntimeError("源数据缺少地理转换信息（GeoTransform）")

    # 设置 Warp 重采样选项
    warp_options = gdal.WarpOptions(
        dstSRS=dst_wkt,
        resampleAlg=resampling,
        format='MEM',
        multithread=True
    )

    # 尝试执行重投影
    dst_dataset = gdal.Warp('', src_dataset, options=warp_options)

    if dst_dataset is None:
        raise RuntimeError("重投影失败，请检查输入数据或坐标系设置。")

    print("投影转换完成！开始收集信息...")

    # 构建返回信息
    info = {
        "width": dst_dataset.RasterXSize,
        "height": dst_dataset.RasterYSize,
        "bands": []
    }

    for band_idx in range(1, dst_dataset.RasterCount + 1):
        band = dst_dataset.GetRasterBand(band_idx)
        stats = band.GetStatistics(True, True)
        band_info = {
            "band": band_idx,
            "min": stats[0],
            "max": stats[1]
        }
        info["bands"].append(band_info)

    return dst_dataset, info

def save_info_to_json(info_dict, bounds_dict, output_path):
    """
    将信息字典和边界字典合并后保存为 output_path/data.json 文件

    参数:
        info_dict: reproject_dataset 返回的元信息字典（含尺寸和波段统计）
        bounds_dict: get_dataset_bounds_in_4326_dict 返回的边界字典（经纬度）
        output_path: 目录路径，例如 'output/'，将在其中生成 'data.json'
    """
    # 合并为一个 JSON 对象
    combined = {
        "dimensions": {
            "width": info_dict.get("width"),
            "height": info_dict.get("height")
        },
        "bands": info_dict.get("bands", []),
        "bounds_4326": bounds_dict
    }

    # 创建目录（如不存在）
    os.makedirs(output_path, exist_ok=True)

    # 拼接成完整文件路径
    json_path = os.path.join(output_path, 'data.json')

    # 写入 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    print(f"信息已保存为 JSON 文件: {json_path}")

def downsample_dataset(dataset, pixel_size, target_resolution=20, no_data_value=-9999):
    """
    对GDAL数据集进行下采样，从原始分辨率下采样到指定分辨率
    使用向量化操作和多线程优化性能
    
    参数:
        dataset: 输入的GDAL数据集
        pixel_size: 原始像素大小（米）
        target_resolution: 目标分辨率（米），默认20米
        no_data_value: 无数据值，默认-9999
    
    返回:
        下采样后的GDAL数据集
    """
    print(f"开始下采样，目标分辨率: {target_resolution}米...")
    
    # 获取原始数据集信息
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    original_width = dataset.RasterXSize
    original_height = dataset.RasterYSize
    band_count = dataset.RasterCount
    
    # 使用传入的像素大小
    original_pixel_size = pixel_size
    
    print(f"原始数据集: {original_width}x{original_height}, 像素大小: {original_pixel_size}米")
    
    # 计算采样窗口大小
    sample_window_size = int(target_resolution / original_pixel_size)
    
    print(f"采样窗口大小: {sample_window_size}x{sample_window_size}")
    
    # 计算新的数据集尺寸
    new_width = original_width // sample_window_size
    new_height = original_height // sample_window_size
    
    print(f"下采样后尺寸: {new_width}x{new_height}")
    
    # 创建新的数据集
    driver = gdal.GetDriverByName('MEM')
    downsampled_dataset = driver.Create('', new_width, new_height, band_count, gdal.GDT_Float32)
    
    # 设置新的地理变换参数
    new_geotransform = [
        geotransform[0],  # 左上角X坐标保持不变
        target_resolution,  # 新的像素宽度
        geotransform[2],  # 旋转参数
        geotransform[3],  # 左上角Y坐标保持不变
        geotransform[4],  # 旋转参数
        -target_resolution  # 新的像素高度（负值）
    ]
    downsampled_dataset.SetGeoTransform(new_geotransform)
    downsampled_dataset.SetProjection(projection)
    
    def process_band_optimized(band_idx):
        """优化的波段处理函数，使用向量化操作"""
        # 获取原始波段
        original_band = dataset.GetRasterBand(band_idx)
        original_data = original_band.ReadAsArray().astype(np.float32)
        
        # 创建新的波段
        new_band = downsampled_dataset.GetRasterBand(band_idx)
        new_band.SetNoDataValue(float(no_data_value))  # 确保是 float 类型
        new_band.SetDescription(original_band.GetDescription())
        
        # 裁剪数据到能整除窗口大小的尺寸
        crop_height = (original_height // sample_window_size) * sample_window_size
        crop_width = (original_width // sample_window_size) * sample_window_size
        cropped_data = original_data[:crop_height, :crop_width]
        
        # 重塑数组以便于批量处理
        # 形状: (new_height, sample_window_size, new_width, sample_window_size)
        reshaped = cropped_data.reshape(
            new_height, sample_window_size,
            new_width, sample_window_size
        )
        
        # 转换为 (new_height, new_width, sample_window_size, sample_window_size)
        reshaped = reshaped.transpose(0, 2, 1, 3)
        
        # 重塑为 (new_height, new_width, window_size^2)
        window_data = reshaped.reshape(new_height, new_width, -1)
        
        # 创建掩码标识有效数据
        valid_mask = window_data != no_data_value
        
        # 计算每个窗口的有效数据数量
        valid_counts = np.sum(valid_mask, axis=2)
        
        # 使用掩码数组计算平均值
        window_data_masked = np.where(valid_mask, window_data, 0)
        window_sums = np.sum(window_data_masked, axis=2)
        
        # 计算平均值，避免除零
        new_data = np.full((new_height, new_width), no_data_value, dtype=np.float32)
        has_valid = valid_counts > 0
        new_data[has_valid] = window_sums[has_valid] / valid_counts[has_valid]
        
        # 写入新波段
        new_band.WriteArray(new_data)
        new_band.FlushCache()
        
        processed_pixels = np.sum(has_valid)
        return band_idx, processed_pixels, new_height * new_width
    
    # 使用多线程处理波段（但GDAL不是线程安全的，所以顺序处理）
    print("正在处理波段...")
    for band_idx in range(1, band_count + 1):
        band_num, processed_pixels, total_pixels = process_band_optimized(band_idx)
        print(f"波段 {band_num} 处理完成，有效像素: {processed_pixels}/{total_pixels}")
    
    print(f"下采样完成！新数据集尺寸: {new_width}x{new_height}, 分辨率: {target_resolution}米")
    
    return downsampled_dataset

def process_dem_to_image_from_datasets(input_dem_path: str, output_path: str, file_suffix: str = "", 
                                      transform_crs: bool = True, source_epsg: int = 2326, target_epsg: int = 3857):
    """
    从DEM文件处理数据到图像，并输出尺寸与统计信息
    """
    
    print("开始处理DEM文件...")
    start_time = time.time()
    
    try:
        # 打开并初始化DEM文件
        dataset = gdal.Open(input_dem_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"无法打开DEM文件: {input_dem_path}")
        geo_transform = dataset.GetGeoTransform()
        pixel_size = int(float(geo_transform[1])+0.5)
        downsampled_dataset = downsample_dataset(dataset, pixel_size, target_resolution=20, no_data_value=-9999)
        dst_dataset,info = reproject_dataset(downsampled_dataset)

        bound = get_dataset_bounds(dst_dataset)

        save_info_to_json(info, bound, output_path)
       
        # 从数据集读取DEM数据
        dem_data = dst_dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
        # 将-9999无数据值替换为0，避免处理错误
        dem_data[dem_data == -9999] = 0.0

        if dem_data is None:
            raise RuntimeError("无法从数据集读取DEM数据")
    
        # 处理地形数据
        norm_data, min_height, max_height = process_terrain_data(dem_data)
        
        # 创建DEM RGBA图像
        image = create_dem_rgba_image(norm_data)
        
        # 保存图像，使用file_suffix作为文件名后缀
        if file_suffix:
            output_dem_path = os.path.join(output_path, f'dem_{file_suffix}.png')
        else:
            output_dem_path = os.path.join(output_path, 'dem.png')
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_dem_path), exist_ok=True)
        
        save_image(image, output_dem_path)
        
        if dataset is not None:
            dataset.FlushCache()
            dataset = None
        if downsampled_dataset is not None:
            downsampled_dataset.FlushCache()
            downsampled_dataset = None
        if dst_dataset is not None:
            dst_dataset.FlushCache()
            dst_dataset = None
        
        # 输出统计信息到日志
        
        processing_time = time.time() - start_time
        print(f"DEM文件处理完成，耗时: {processing_time:.2f} 秒")
        
        return bound, info
        
    except Exception as e:
        print(f"处理DEM文件时出错: {e}")
        # 确保资源被清理
        if 'dataset' in locals():
            dataset = None
        raise

 
# dem_input_path = "./HK_dem/Digital Terrain Model.tif"
# dem_output_path = "./dem/"
# bound, info = process_dem_to_image_from_datasets(dem_input_path, dem_output_path)
# print(f"字段统计信息: {info}")
# print(f"图片尺寸: {bound}")