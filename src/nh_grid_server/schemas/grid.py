import json
import math
import numpy as np
from pathlib import Path
from pydantic import BaseModel, field_validator
from .base import BaseResponse
from .schema import ProjectSchema
from ..core.config import settings, APP_CONTEXT
from .project import ProjectMeta, SubprojectMeta

class GridMeta(BaseModel):
    """Meta information for a specific grid resource"""
    name: str # name of the grid
    epsg: int # EPSG code for the grid
    subdivide_rules: list[tuple[int, int]] # rules for subdividing the grid
    bounds: tuple[float, float, float, float] # [ min_lon, min_lat, max_lon, max_lat ]
    
    @staticmethod
    def from_subproject(project_name: str, subproject_name: str):
        """Create a GridMeta instance from a subproject"""
        
        project_dir = Path(settings.PROJECT_DIR, project_name)
        subproject_dir = project_dir / subproject_name
        project_meta_file = project_dir / settings.GRID_PROJECT_META_FILE_NAME
        subproject_meta_file = subproject_dir / settings.GRID_SUBPROJECT_META_FILE_NAME
        
        try:
            # Get bounds from subproject meta file
            with open(subproject_meta_file, 'r') as f:
                subproject_data = json.load(f)
            subproject_meta = SubprojectMeta(**subproject_data)
            bounds = subproject_meta.bounds
            
            # Get grid info from project meta file
            with open(project_meta_file, 'r') as f:
                project_data = json.load(f)
            project_meta = ProjectMeta(**project_data)
            
            schema_name = project_meta.schema_name
            schema_file = Path(settings.SCHEMA_DIR, f'{schema_name}.json')
            
            with open(schema_file, 'r') as f:
                schema_data = json.load(f)
            schema_meta = ProjectSchema(**schema_data)
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
                name=subproject_name,
                epsg=epsg,
                subdivide_rules=subdivide_rules,
                bounds=bounds
            )
            
        except Exception as e:
            raise ValueError(f'Failed to create grid meta information: {str(e)}')
    
    @staticmethod
    def from_context():
        """Create a GridMeta instance from a subproject"""

        project_name = APP_CONTEXT['current_project']
        subproject_name = APP_CONTEXT['current_subproject']
        return GridMeta.from_subproject(project_name, subproject_name)

class MultiGridInfo(BaseModel):
    levels: list[int]
    global_ids: list[int]
    
    @field_validator('levels')
    def check_levels(cls, v):
        if len(v) == 0:
            raise ValueError('Levels cannot be empty')
        return v
    
    @field_validator('global_ids')
    def check_global_ids(cls, v):
        if len(v) == 0:
            raise ValueError('Global IDs cannot be empty')
        return v
    
    def to_bytes_dict(self):
        """Convert the grid information to a binary-efficient dictionary."""
        
        level_arr = np.array(self.levels, dtype=np.int8)
        global_id_arr = np.array(self.global_ids, dtype=np.int32)
        
        grid_num = len(self.levels)
        level_bytes = level_arr.tobytes()
        global_id_bytes = global_id_arr.tobytes()
        
        level_str = ''.join(chr(x) for x in level_bytes)
        global_id_str = ''.join(chr(x) for x in global_id_bytes)
        
        return {
            'grid_num': grid_num,
            'levels_bytes': level_str,
            'global_ids_bytes': global_id_str,
        }
    
    def combine_bytes(self):
        """
        Combine the grid information into a single bytes object
        
        Format: [4 bytes for length, followed by level bytes, followed by global id bytes]
        """
        
        level_bytes = np.array(self.levels, dtype=np.int8).tobytes()
        global_id_bytes = np.array(self.global_ids, dtype=np.int32).tobytes()
        
        level_length = len(level_bytes).to_bytes(4, byteorder='little')
        padding_size = (4 - (len(level_length) + len(level_bytes)) % 4) % 4
        padding = b'\x00' * padding_size
        
        return level_length + level_bytes + padding + global_id_bytes

class MultiGridInfoResponse(BaseResponse):
    """Standard response schema for grid operations"""
    infos: dict[str, str | int] # bytes representation of MultiGridInfo, { 'grid_num': num, 'levels': b'...', 'global_ids': b'...' }
    
    @field_validator('infos')
    def check_infos(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Infos must be a dictionary')
        return v
    