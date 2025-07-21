import os
import struct
import logging
import c_two as cc
from pathlib import Path

from icrms.isolution import ISolution
from src.nh_grid_server.core.config import settings

logger = logging.getLogger(__name__)

class HydroElement:
    def __init__(self, data: bytes):
        # Unpack index, bounds and edge counts
        index, min_x, min_y, max_x, max_y, left_edge_num, right_edge_num, bottom_edge_num, top_edge_num = struct.unpack('!QddddBBBB', data[:44])
        self.index: int = index
        self.bounds: tuple[float, float, float, float] = (min_x, min_y, max_x, max_y)

        # Unpack edges
        total_edge_num = left_edge_num + right_edge_num + bottom_edge_num + top_edge_num
        edge_coords_types = '!' + 'Q' * total_edge_num
        edges: list[int] = list(struct.unpack(edge_coords_types, data[44:]))
        
        # Calculate edge starts
        left_edge_start = 0
        right_edge_start = left_edge_num
        bottom_edge_start = right_edge_start + right_edge_num
        top_edge_start = bottom_edge_start + bottom_edge_num
        
        # Extract edges
        self.left_edges: list[int] = edges[left_edge_start:right_edge_start]
        self.right_edges: list[int] = edges[right_edge_start:bottom_edge_start]
        self.bottom_edges: list[int] = edges[bottom_edge_start:top_edge_start]
        self.top_edges: list[int] = edges[top_edge_start:]

        # Default attributes (can be modified later)
        self.altitude = -9999.0     # placeholder for altitude
        self.type = 0               # default element type (0 for hydro default)
    
    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.bounds[0] + self.bounds[2]) / 2.0,  # center x
            (self.bounds[1] + self.bounds[3]) / 2.0,  # center y
            self.altitude,                            # center z
        )
    
    @property
    def ne(self) -> list[int | float]:
        return [
            self.index,                                     # element index
            len(self.left_edges),                           # number of left edges
            len(self.right_edges),                          # number of right edges
            len(self.bottom_edges),                         # number of bottom edges
            len(self.top_edges),                            # number of top edges
            *self.left_edges,                               # left edge indices
            *self.right_edges,                              # right edge indices
            *self.bottom_edges,                             # bottom edge indices
            *self.top_edges,                                # top edge indices
            *self.center,                                   # center coordinates (x, y, z)
            self.type,                                      # element type
        ]

class HydroSide:
    def __init__(self, data: bytes):
        # Unpack index, direction, bounds and adjacent grid indices
        index, direction, min_x, min_y, max_x, max_y, grid_index_a, grid_index_b = struct.unpack('!QBddddQQ', data)
        self.index = index
        self.direction = direction
        self.bounds = (min_x, min_y, max_x, max_y)
        self.grid_index_a = grid_index_a
        self.grid_index_b = grid_index_b
        
        # Default attributes (can be modified later)
        self.altitude = -9999.0  # placeholder for altitude
        self.type = 0            # default side type (0 for hydro default)
    
    @property
    def length(self) -> float:
        return (self.bounds[2] - self.bounds[0]) if self.direction == 1 else (self.bounds[3] - self.bounds[1])
    
    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.bounds[0] + self.bounds[2]) / 2.0,  # center x
            (self.bounds[1] + self.bounds[3]) / 2.0,  # center y
            self.altitude,                            # center z
        )
    
    @property
    def ns(self) -> list[int | float]:
        left_grid_index, right_grid_index, bottom_grid_index, top_grid_index = 0, 0, 0, 0
        if self.direction == 0: # vertical side
            left_grid_index = self.grid_index_a if self.grid_index_a is not None else 0
            right_grid_index = self.grid_index_b if self.grid_index_b is not None else 0
        else: # horizontal side
            top_grid_index = self.grid_index_a if self.grid_index_a is not None else 0
            bottom_grid_index = self.grid_index_b if self.grid_index_b is not None else 0
            
        return [
            self.index,             # side index
            self.direction,         # direction (0 for vertical, 1 for horizontal)
            left_grid_index,        # left grid index (1-based)
            right_grid_index,       # right grid index (1-based)
            bottom_grid_index,      # bottom grid index (1-based)
            top_grid_index,         # top grid index (1-based)
            self.length,            # length of the side
            *self.center,           # center coordinates (x, y, z)
            self.type,              # side type
        ]

@cc.iicrm
class Solution(ISolution):
    def __init__(self, name: str, env: dict):
        self.name = name
        self.path = Path(f'{settings.SOLUTION_DIR}{self.name}')
        self.env = env

        # Create solution directory
        self.path.mkdir(parents=True, exist_ok=True)
        # # Create ref json file
        # ref_path = self.path / 'ref.json'
        # with open(ref_path, 'w', encoding='utf-8') as f:
        #     json.dump(body.model_dump(), f, ensure_ascii=False, indent=4)

    def clone_env(self) -> dict:
        env_data = {}
        for key, value in self.env.items():
            if isinstance(value, str) and os.path.isfile(value):
                try:
                    with open(value, 'r', encoding='utf-8') as f:
                        content = f.readlines()
                except UnicodeDecodeError:
                    with open(value, 'rb') as f:
                        content = f.read()
                env_data[key] = {
                    'file_name': os.path.basename(value),
                    'content': content
                }
            else:
                env_data[key] = value
        return env_data
    
    def get_env(self) -> dict:
        return self.env

    def terminate(self) -> None:
        # Do something need to be saved
        pass