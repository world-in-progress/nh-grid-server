import os
import time
import json
import shutil
import logging
import zipfile
import c_two as cc
import multiprocessing
from pathlib import Path
from crms.grid import Grid
from crms.raster import Raster
from crms.common import Common
from crms.treeger import Treeger
from persistence.helpers.DemGeneration import process_dem_to_image_from_datasets
from fastapi import Request  # 添加这一行

from icrms.isolution import ISolution
from src.nh_grid_server.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SRC_CRS = "EPSG:2326"

@cc.iicrm
class Solution(ISolution):
    def __init__(self, name: str, model_type: str, env: dict, action_types: list[str]):
        self.name = name
        self.model_type = model_type
        self.env = env
        self.action_types = action_types
        
        self.path = Path(f'{settings.SOLUTION_DIR}{self.name}')
        self.env_path = self.path / 'env'
        self.render_path = self.path / 'render'
        self.human_actions_path = self.path / 'actions' / 'human_actions'
        self.model_env_path = self.path / 'model_env.json'

        self.path.mkdir(parents=True, exist_ok=True)
        self.env_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)
        self.human_actions_path.mkdir(parents=True, exist_ok=True)

    def get_action_types(self) -> list[str]:
        return self.action_types
    
    def add_human_action(self, action_type: str, params: dict) -> str:
        action_id = str(int(time.time() * 1000))
        action_path = self.human_actions_path / f'{action_id}.json'
        
        # 获取params数据并去掉action_type字段
        params_data = params.model_dump()
        params_data.pop('action_type', None)  # 安全地移除action_type字段
            
        with open(action_path, 'w', encoding='utf-8') as f:
            json.dump({
                'action_type': action_type,
                'params': params_data
            }, f, ensure_ascii=False, indent=4)
        return action_id

    def update_human_action(self, action_id, params):
        action_path = self.human_actions_path / f'{action_id}.json'
        if not action_path.exists():
            raise FileNotFoundError(f'Action file {action_path} does not exist.')
        
        # 获取params数据并去掉action_type字段
        params_data = params.model_dump()
        params_data.pop('action_type', None)  # 安全地移除action_type字段
        
        with open(action_path, 'w', encoding='utf-8') as f:
            json.dump({
                'action_type': params.action_type,
                'params': params_data
            }, f, ensure_ascii=False, indent=4)

    def delete_human_action(self, action_id):
        action_path = self.human_actions_path / f'{action_id}.json'
        if action_path.exists():
            action_path.unlink()
        else:
            logger.warning(f'Action file {action_path} does not exist.')
    
    def get_human_actions(self) -> list[dict]:
        actions = []
        try:
            # 检查actions目录是否存在
            if not self.human_actions_path.exists():
                logger.warning(f'Actions path {self.human_actions_path} does not exist')
                return actions
            
            # 遍历actions目录下的所有JSON文件
            for action_file in self.human_actions_path.glob('*.json'):
                try:
                    with open(action_file, 'r', encoding='utf-8') as f:
                        action_data = json.load(f)
                        # 添加action_id（从文件名提取）
                        action_id = action_file.stem  # 去掉.json后缀
                        action_data['action_id'] = action_id
                        actions.append(action_data)
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f'Failed to read action file {action_file}: {str(e)}')
                    continue
            
            # 按action_id排序（时间戳顺序）
            actions.sort(key=lambda x: x.get('action_id', '0'))
            
        except Exception as e:
            logger.error(f'Failed to get human actions: {str(e)}')
        
        return actions

    def ne_sampling(self, dem_crm, lum_crm, grid_list) -> dict:
        try:
            ne_list = []
            for grid in grid_list:
                center = grid.center
                x = center[0]
                y = center[1]
                grid.altitude = dem_crm.sampling(x, y, src_crs=DEFAULT_SRC_CRS)
                grid.type = lum_crm.sampling(x, y, src_crs=DEFAULT_SRC_CRS)
                ne_list.append(grid.ne)
                print(f"Grid {grid.index} - Altitude: {grid.altitude}, Type: {grid.type}")

            ne_path = self.env_path / 'ne.txt'

            with open(ne_path, 'w', encoding='utf-8') as f:
                for ne in ne_list:
                    if isinstance(ne, (list, tuple)):
                        # 将列表元素用空格或逗号分隔
                        f.write(','.join(map(str, ne)) + '\n')
                    else:
                        f.write(str(ne) + '\n')

            return {"status": True, "message": "ne.txt"}
        except Exception as e:
            logger.error(f'Failed to perform sampling: {str(e)}')
            return {"status": False, "message": str(e)}

    def ns_sampling(self, dem_crm, lum_crm, edge_list) -> dict:
        try:
            ns_list = []
            for edge in edge_list:
                center = edge.center
                x = center[0]
                y = center[1]
                edge.altitude = dem_crm.sampling(x, y, src_crs=DEFAULT_SRC_CRS)
                edge.type = lum_crm.sampling(x, y, src_crs=DEFAULT_SRC_CRS)
                ns_list.append(edge.ns)
                print(f"Edge {edge.index} - Altitude: {edge.altitude}, Type: {edge.type}")

            ns_path = self.env_path / 'ns.txt'

            with open(ns_path, 'w', encoding='utf-8') as f:
                for ns in ns_list:
                    if isinstance(ns, (list, tuple)):
                        # 将列表元素用空格或逗号分隔
                        f.write(','.join(map(str, ns)) + '\n')
                    else:
                        f.write(str(ns) + '\n')

            return {"status": True, "message": "ns.txt"}
        except Exception as e:
            logger.error(f'Failed to perform sampling: {str(e)}')
            return {"status": False, "message": str(e)}

    def package(self) -> dict:
        try:
            treeger = Treeger()
            grid_node_key = self.env.get('grid_node_key')
            dem_node_key = self.env.get('dem_node_key')
            lum_node_key = self.env.get('lum_node_key')
            rainfall_node_key = self.env.get('rainfall_node_key')
            gate_node_key = self.env.get('gate_node_key')
            tide_node_key = self.env.get('tide_node_key')
            inp_node_key = self.env.get('inp_node_key')

            grid_crm = treeger.trigger(grid_node_key, Grid)
            dem_crm = treeger.trigger(dem_node_key, Raster)
            lum_crm = treeger.trigger(lum_node_key, Raster)
            rainfall_crm = treeger.trigger(rainfall_node_key, Common)
            gate_crm = treeger.trigger(gate_node_key, Common)
            tide_crm = treeger.trigger(tide_node_key, Common)
            inp_crm = treeger.trigger(inp_node_key, Common)

            # 1. ne and ns sampling
            grid_list = grid_crm.parse_grid_records()
            edge_list = grid_crm.parse_edge_records()
            ne_sampling_result = self.ne_sampling(dem_crm, lum_crm, grid_list)
            ns_sampling_result = self.ns_sampling(dem_crm, lum_crm, edge_list)
            if not ne_sampling_result.get('status', True):
                logger.error(f'NE sampling failed: {ne_sampling_result.get("message", "")}')
            if not ns_sampling_result.get('status', True):
                logger.error(f'NS sampling failed: {ns_sampling_result.get("message", "")}')

            # 2. copy common files
            rainfall_filename = rainfall_crm.copy_to(self.env_path).get('message', 'rainfall.txt')
            gate_filename = gate_crm.copy_to(self.env_path).get('message', 'gate.txt')
            tide_filename = tide_crm.copy_to(self.env_path).get('message', 'tide.txt')
            inp_filename = inp_crm.copy_to(self.env_path).get('message', 'inp.txt')

            # 3. copy render files
            render_path = Path(os.path.join(settings.PERSISTENCE_DIR, 'render'))
            shutil.copytree(render_path, self.render_path, dirs_exist_ok=True)

            # 4. generate render resource
            process = multiprocessing.Process(
                target=process_dem_to_image_from_datasets,
                kwargs={
                    'input_dem_path': dem_crm.get_cog_tif(),
                    'output_path': self.render_path / 'static' / 'dem/'
                }
            )
            process.start()
            process.join()

            # 5. create package
            package_path = self.path / f'{self.name}_package.zip'
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as package_zip:
                # 添加solution目录中的所有文件和文件夹
                for root, dirs, files in os.walk(self.path):
                    root_path = Path(root)
                    
                    # 跳过生成的压缩包文件本身
                    if root_path == self.path and f'{self.name}_package.zip' in files:
                        files.remove(f'{self.name}_package.zip')
                    
                    # 添加所有文件
                    for file in files:
                        file_path = root_path / file
                        arcname = file_path.relative_to(self.path)
                        package_zip.write(file_path, arcname)

            logger.info(f'Package created: {package_path}')

            self.model_env = {
                "ne": ne_sampling_result.get('message', 'ne.txt'),
                "ns": ns_sampling_result.get('message', 'ns.txt'),
                "rainfall": rainfall_filename,
                "gate": gate_filename,
                "tide": tide_filename,
                "inp": inp_filename
            }

            
            with open(self.model_env_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_env, f, ensure_ascii=False, indent=4)

            return {"status": True, "message": "Model environment saved successfully"}
        except Exception as e:
            logger.error(f'Failed to package solution: {str(e)}')
            return {"status": False, "message": str(e)}

    def delete_solution(self) -> None:
        """
        删除解决方案
        :return: None
        """
        ...
        # 删除解决方案目录
        if self.path.exists():
            shutil.rmtree(self.path)
            logger.info(f'Solution directory {self.path} deleted successfully')
        else:
            logger.warning(f'Solution directory {self.path} does not exist')

    def get_solution(self) -> dict:
        """
        获取解决方案
        :return: 解决方案字典
        """
        solution_data = {
            "name": self.name,
            "model_type": self.model_type,
            "env": self.env,
            "action_types": self.action_types
        }
        return solution_data

    def get_terrain_data(self, base_url: str = None) -> dict:
        """
        获取地形数据字典
        :param base_url: 基础URL，用于构建完整的地形图URL
        :return: 地形数据字典
        """
        try:
            data = {}
            data_path = self.render_path / 'static' / 'dem' / 'data.json'
            
            # 构建完整的URL
            if base_url:
                dem_url = f"{base_url}/solutions/{self.name}/render/static/dem/dem.png"
            else:
                # 如果没有base_url，使用相对路径作为后备
                dem_url = f"/solutions/{self.name}/render/static/dem/dem.png"
            
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning(f'Terrain data file {data_path} does not exist')
            # 安全地获取 dimensions 数据
            dimensions = data.get('dimensions', {})
            if dimensions is None:
                dimensions = {}
            
            # 安全地获取 bands 数据
            bands = data.get('bands', [])
            if bands is None or len(bands) == 0:
                bands = [{'min': 0, 'max': 0}]
            
            terrain_data = {
                "terrainMap": dem_url,
                "terrainMapSize": [dimensions.get('width', 0), dimensions.get('height', 0)],
                "terrainHeightMin": bands[0].get('min', 0),
                "terrainHeightMax": bands[0].get('max', 0),
            }
            return terrain_data
            
        except Exception as e:
            logger.error(f'Failed to get terrain data: {str(e)}')
            # 返回默认的地形数据结构
            if base_url:
                dem_url = f"{base_url}/solutions/{self.name}/render/static/dem/dem.png"
            else:
                dem_url = f"/solutions/{self.name}/render/static/dem/dem.png"
            
            return {
                "terrainMap": dem_url,
                "terrainMapSize": [0, 0],
                "terrainHeightMin": 0,
                "terrainHeightMax": 0,
            }

    # From Model Server
    def clone_package(self) -> dict:
        """
        获取解决方案的压缩包供其他服务访问
        :return: 包含压缩包信息和数据的字典
        """
        try:
            package_path = self.path / f'{self.name}_package.zip'
            
            if not package_path.exists():
                logger.warning(f'Package file {package_path} does not exist')
                return {
                    "status": False,
                    "message": f"Package file for solution '{self.name}' not found",
                    "package_data": None
                }
            
            # 读取压缩包二进制数据
            with open(package_path, 'rb') as package_file:
                package_data = package_file.read()
            
            return {
                "status": True,
                "message": "Package ready for download",
                "package_data": package_data
            }
        except Exception as e:
            logger.error(f'Failed to prepare package for download: {str(e)}')
            return {
                "status": False,
                "message": f"Failed to prepare package: {str(e)}",
                "package_data": None
            }

    def get_model_env(self) -> dict:
        if self.model_env_path.exists():
            with open(self.model_env_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def terminate(self) -> None:
        # Do something need to be saved
        pass