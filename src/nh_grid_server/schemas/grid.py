import json
import math
import numpy as np
from pathlib import Path
from .patch import PatchMeta
from .base import BaseResponse
from .schema import GridSchema
from pydantic import BaseModel, field_validator
from ..core.config import settings, APP_CONTEXT

class GridMeta(BaseModel):
    """Meta information for a specific grid resource"""
    name: str # name of the grid
    epsg: int # EPSG code for the grid
    description: str = '' # description of the grid, optional
    subdivide_rules: list[tuple[int, int]] # rules for subdividing the grid
    bounds: tuple[float, float, float, float] # [ min_lon, min_lat, max_lon, max_lat ]
    
    @staticmethod
    def from_patch(schema_name: str, patch_name: str):
        """Create a GridMeta instance from a patch"""
        
        schema_file = Path(settings.GRID_SCHEMA_DIR, schema_name, 'schema.json')
        patch_meta_file = Path(settings.GRID_SCHEMA_DIR, schema_name, 'patches', patch_name, settings.GRID_PATCH_META_FILE_NAME)

        try:
            # Get bounds from patch meta file
            with open(patch_meta_file, 'r') as f:
                patch_data = json.load(f)
            patch_meta = PatchMeta(**patch_data)
            bounds = patch_meta.bounds
            
            with open(schema_file, 'r') as f:
                schema_data = json.load(f)
            schema_meta = GridSchema(**schema_data)
            epsg = schema_meta.epsg
            grid_info = schema_meta.grid_info
            first_size = grid_info[0]
    
            # Calculate subdivide rules
            subdivide_rules: list[list[int]] = [
                [
                    int(math.ceil((bounds[2] - bounds[0]) / first_size[0])),
                    int(math.ceil((bounds[3] - bounds[1]) / first_size[1])),
                ]
            ]
            for i in range(len(grid_info) - 1):
                level_a = grid_info[i]
                level_b = grid_info[i + 1]
                subdivide_rules.append(
                    [
                        int(level_a[0] / level_b[0]),
                        int(level_a[1] / level_b[1]),
                    ]
                )
            subdivide_rules.append([1, 1])
            
            return GridMeta(
                name=patch_name,
                epsg=epsg,
                description=patch_meta.description,
                subdivide_rules=subdivide_rules,
                bounds=bounds
            )
            
        except Exception as e:
            raise ValueError(f'Failed to create grid meta information: {str(e)}')
    
    @staticmethod
    def from_context():
        """Create a GridMeta instance from a patch"""

        project_name = APP_CONTEXT['current_project']
        patch_name = APP_CONTEXT['current_patch']
        return GridMeta.from_patch(project_name, patch_name)
    
class MultiGridInfo(BaseModel):
    levels: list[int]
    global_ids: list[int]
    
    def combine_bytes(self):
        """
        Combine the grid information into a single bytes object
        
        Format: [4 bytes for length, followed by level bytes, followed by global id bytes]
        """
        
        level_bytes = np.array(self.levels, dtype=np.uint8).tobytes()
        global_id_bytes = np.array(self.global_ids, dtype=np.uint32).tobytes()
        
        level_length = len(level_bytes).to_bytes(4, byteorder='little')
        padding_size = (4 - (len(level_length) + len(level_bytes)) % 4) % 4
        padding = b'\x00' * padding_size
        
        return level_length + level_bytes + padding + global_id_bytes
    
    @staticmethod
    def from_bytes(data: bytes):
        """
        Create a MultiGridInfo instance from bytes data
        
        The data format is:
        - First 4 bytes: length of the level bytes
        - Next N bytes: level bytes
        - Padding to make the total length a multiple of 4
        - Remaining bytes: global id bytes
        """
        
        if len(data) < 8:
            raise ValueError('Data is too short to contain valid MultiGridInfo')
        
        level_length = int.from_bytes(data[:4], byteorder='little')
        level_bytes = data[4:4 + level_length]
        global_id_bytes = data[4 + level_length + (4 - (level_length % 4)) % 4:]
        
        levels = list(np.frombuffer(level_bytes, dtype=np.uint8))
        global_ids = list(np.frombuffer(global_id_bytes, dtype=np.uint32))
        
        return MultiGridInfo(levels=levels, global_ids=global_ids)

class MultiGridInfoResponse(BaseResponse):
    """Standard response schema for grid operations"""
    infos: dict[str, str | int] # bytes representation of MultiGridInfo, { 'grid_num': num, 'levels': b'...', 'global_ids': b'...' }
    
    @field_validator('infos')
    def check_infos(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Infos must be a dictionary')
        return v
    