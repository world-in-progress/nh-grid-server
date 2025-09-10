import os
import time
import math
import json
import mmap
import struct
import atexit
import logging
import c_two as cc
import numpy as np
import multiprocessing as mp

from pathlib import Path
from enum import IntEnum
from typing import Callable
from functools import partial
from pydantic import BaseModel

from crms.patch import Patch
from icrms.igrid import IGrid
from crms.treeger import Treeger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENARIO_META_PATH = os.getenv('SCENARIO_META_PATH')

WORKER_PATCH_OBJ = None
WORKER_MMAP_OBJ = None
WORKER_FILE_HANDLE = None

EDGE_CODE_INVALID = -1
class EdgeCode(IntEnum):
    NORTH = 0b00  # 0
    WEST  = 0b01  # 1
    SOUTH = 0b10  # 2
    EAST  = 0b11  # 3
    
TOGGLE_EDGE_CODE_MAP = {
    EdgeCode.NORTH: EdgeCode.SOUTH,
    EdgeCode.WEST: EdgeCode.EAST,
    EdgeCode.SOUTH: EdgeCode.NORTH,
    EdgeCode.EAST: EdgeCode.WEST
}

ADJACENT_CHECK_NORTH = lambda local_id, sub_width, sub_height: local_id < sub_width
ADJACENT_CHECK_EAST = lambda local_id, sub_width, sub_height: local_id % sub_width == 0
ADJACENT_CHECK_WEST = lambda local_id, sub_width, sub_height: local_id % sub_width == sub_width - 1
ADJACENT_CHECK_SOUTH = lambda local_id, sub_width, sub_height: local_id >= sub_width * (sub_height - 1)

class PatchInfo(BaseModel):
    node_key: str
    treeger_address: str

class PatchInfoList(BaseModel):
    patches: list[PatchInfo]

class Overview:
    def __init__(self, size):
        self.size = size
        self.data = bytearray((size + 7) // 8)
    
    @staticmethod
    def create(data: bytearray):
        instance = Overview(0)
        instance.size = len(data) * 8
        instance.data = data
        return instance

    def set_value(self, index, value):
        if index < 0 or index >= self.size:
            raise IndexError('3Index out of bounds')
        
        byte_index = index // 8
        bit_index = index % 8
        if value:
            self.data[byte_index] |= (1 << bit_index)
        else:
            self.data[byte_index] &= ~(1 << bit_index)

    def get_value(self, index):
        if index < 0 or index >= self.size:
            raise IndexError('4Index out of bounds')

        byte_index = index // 8
        bit_index = index % 8
        return (self.data[byte_index] >> bit_index) & 1
    
    @property
    def binary_sequence(self) -> str:
        binary_string_parts = []
        for i in range(self.size):
            binary_string_parts.append(str(self.get_value(i)))
        return ''.join(binary_string_parts)

class GridCache:
    class _ArrayView:
        def __init__(self, parent: 'GridCache'):
            self._parent = parent
        
        def __getitem__(self, index: int) -> tuple[int, int]:
            if index < 0 or index >= len(self):
                raise IndexError('5Index out of bounds')
            return self._parent._decode_at_index(index)

        def __len__(self) -> int:
            return self._parent._len
            
        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

    def __init__(self, data: bytes):
        if len(data) % 9 != 0:
            raise ValueError('Data must be a multiple of 9 bytes long')
        self.data = data
        self._len = len(self.data) // 9
        
        self.array = self._ArrayView(self)
        self.map = {index : i for i, index in enumerate(self.array)}

        self.fract_coords: list[tuple[list[int], list[int], list[int], list[int]]] = []

        self.edges: list[list[set[int]]] = [[set() for _ in range(4)] for _ in range(self._len)]
        self.neighbours: list[list[set[int]]] = [[set() for _ in range(4)] for _ in range(self._len)]

    def __len__(self) -> int:
        return self._len
    
    def __repr__(self) -> str:
        return f'<GridBytes with {self._len} items>'
    
    def _decode_at_index(self, index: int) -> tuple[int, int]:
        start = index * 9
        subdata = self.data[start : start + 9]
        return struct.unpack('!BQ', subdata)

    def has_grid(self, level: int, global_id: int) -> bool:
        return (level, global_id) in self.map

    def slice_grids(self, start_index: int, length: int) -> bytes:
        if start_index < 0 or start_index > self._len:
            raise IndexError('1Index out of bounds')
        start = start_index * 9
        end = min(start + length * 9, self._len * 9)
        return self.data[start:end]
    
    def slice_edges(self, start_index: int, length: int) -> bytes:
        if start_index < 0 or start_index > self._len:
            raise IndexError('2Index out of bounds')
        end_index = min(start_index + length, self._len)
        return self.edges[start_index : end_index]

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
        
        direction = 1 if self.direction == 1 else 2
            
        return [
            self.index,             # side index
            direction,              # direction (1 for horizontal, 2 for vertical)
            left_grid_index,        # left grid index (1-based)
            right_grid_index,       # right grid index (1-based)
            bottom_grid_index,      # bottom grid index (1-based)
            top_grid_index,         # top grid index (1-based)
            self.length,            # length of the side
            *self.center,           # center coordinates (x, y, z)
            self.type,              # side type
        ]

@cc.iicrm
class Grid(IGrid):
    def __init__(self, schema_path: str, workspace: str):
        self.workspace = Path(workspace)
        self.schema_path = Path(schema_path)
        self.meta_ov_path = self.workspace / 'meta_overview.bin'
        self.grid_record_path = self.workspace / 'grid_records.bin'
        self.edge_record_path = self.workspace / 'edge_records.bin'
        
        # Init workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Init schema info
        schema = json.load(open(self.schema_path, 'r'))
        self.epsg: int = schema['epsg']
        self.grid_info: list[list[float]] = schema['grid_info']
        self.first_level_grid_size: list[float] = self.grid_info[0]
        self.first_level_width = 0
        self.first_level_height = 0
        
        # Init meta overview properties
        self.meta_ov_byte_length = 0
        self.ov_info: list[tuple[list[int], int]] = []
        for i in range(len(self.grid_info)):
            width = int(self.grid_info[0][0] / self.grid_info[i][0])
            height = int(self.grid_info[0][1] / self.grid_info[i][1])
            self.ov_info.append((
                [width, height],
                width * height
            ))
        
        self.ov_bit_length = 0
        for info in self.ov_info:
            _, size = info
            self.ov_bit_length += size
        self.ov_byte_length = (self.ov_bit_length + 7) // 8

        self.ov_offset = [0]
        for info in self.ov_info:
            _, size = info
            self.ov_offset.append(self.ov_offset[-1] + size)

        # Init bounds, subdivide_rules and meta_level_info, which will be updated later
        inf, neg_inf = float('inf'), float('-inf')
        self.bounds = [inf, inf, neg_inf, neg_inf]  # [min_x, min_y, max_x, max_y]
        self.subdivide_rules: list[list[int]] = []
        self.meta_level_info: list[dict[str, int]] = [{'width': 1, 'height': 1}]
        
        # Get all patch info
        with open(self.workspace / 'patches.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.patch_infos = PatchInfoList(**data).patches

        # Initialize cache
        self._edge_index_cache: list[bytes] = []
        self._edge_index_dict: dict[int, bytes] = {}
        self._edge_adj_grids_indices: list[list[int | None]] = [] # for each edge, the list of adjacent grid indices, among [grid_a, grid_b], grid_a must be the north or west grid

    def _create_meta_overview(self, treeger: Treeger):
        # Update bounds
        for patch_info in self.patch_infos:
            patch = treeger.trigger(patch_info.node_key, Patch)
            schema = patch.get_schema()
            self.bounds[0] = min(self.bounds[0], schema.bounds[0])
            self.bounds[1] = min(self.bounds[1], schema.bounds[1])
            self.bounds[2] = max(self.bounds[2], schema.bounds[2])
            self.bounds[3] = max(self.bounds[3], schema.bounds[3])

        # Update subdivide rules
        self.subdivide_rules = [
            [
                int(math.ceil((self.bounds[2] - self.bounds[0]) / self.first_level_grid_size[0])),
                int(math.ceil((self.bounds[3] - self.bounds[1]) / self.first_level_grid_size[1])),
            ]
        ]
        for i in range(len(self.grid_info) - 1):
            level_a = self.grid_info[i]
            level_b = self.grid_info[i + 1]
            self.subdivide_rules.append(
                [
                    int(level_a[0] / level_b[0]),
                    int(level_a[1] / level_b[1]),
                ]
            )
        self.subdivide_rules.append([1, 1])
        
        # Update level info
        for level, rule in enumerate(self.subdivide_rules[:-1]):
            prev_width, prev_height = self.meta_level_info[level]['width'], self.meta_level_info[level]['height']
            self.meta_level_info.append({
                'width': prev_width * rule[0],
                'height': prev_height * rule[1]
            })

        # Update first level width and height
        self.first_level_width = self.meta_level_info[1]['width']
        self.first_level_height = self.meta_level_info[1]['height']
        
        # Create overview
        self.meta_ov_byte_length = self.first_level_width * self.first_level_height * self.ov_byte_length
        
        # Create meta overview file
        with open(self.meta_ov_path, 'wb') as f:
            f.write(b'\x00' * self.meta_ov_byte_length)

    def _process_patch(self, treeger: Treeger, patch_info: PatchInfo):
        patch = treeger.trigger(patch_info.node_key, Patch)
        patch_width = patch.level_info[1]['width']
        patch_height = patch.level_info[1]['height']
        
        batch_args = [
            row_index
            for row_index in range(patch_height)
        ]
        batch_func = partial(
            _process_chunk_overview_worker,
            patch_width=patch_width,
            ov_offset=self.ov_offset,
            ov_bit_length=self.ov_bit_length,
            bounds=self.bounds,
            first_level_grid_size=self.first_level_grid_size,
            first_level_width=self.first_level_width,
            ov_byte_length=self.ov_byte_length,
            meta_level_info=self.meta_level_info,
            grid_info=self.grid_info
        )
        
        num_processes = min(os.cpu_count(), len(batch_args))
        with mp.Pool(processes=num_processes, initializer=_init_chunk_overview_worker, initargs=(patch, self.meta_ov_path)) as pool:
            pool.map(batch_func, batch_args)

    def _find_all_active_grids(self) -> list[int]:
        batch_size = 1000 * self.ov_byte_length
        batch_args = [
            batch_byte_offset
            for batch_byte_offset in range(0, self.meta_ov_byte_length, batch_size)
        ]
        
        batch_func = partial(
            _batch_process_overview_worker,
            batch_size=batch_size,
            ov_byte_length=self.ov_byte_length,
            ov_offset=self.ov_offset,
            meta_level_info=self.meta_level_info,
            subdivide_rules=self.subdivide_rules,
            grid_info=self.grid_info
        )
        
        num_processes = min(os.cpu_count(), len(batch_args))
        with mp.Pool(processes=num_processes, initializer=_init_batch_process_overview_worker, initargs=(self.meta_ov_path,)) as pool:
            active_grids_list = pool.map(batch_func, batch_args)

        active_grid_info: bytearray = bytearray()
        for active_grids in active_grids_list:
            active_grid_info += active_grids
        return active_grid_info
    
    def _get_grid_from_uv(self, level: int, level_width, level_height, u: int, v: int) -> tuple[int, int] | None:
        if level >= len(self.meta_level_info) or level < 0:
            return None
        
        if u < 0 or u >= level_width or v < 0 or v >= level_height:
            return None
        
        global_id = v * level_width + u
        return (level, global_id)
    
    def _get_toggle_edge_code(self, code: int) -> int:
        return TOGGLE_EDGE_CODE_MAP.get(code, EDGE_CODE_INVALID)
    
    def _update_grid_neighbour(
        self, grid_cache: GridCache, 
        grid_level: int, grid_global_id: int, 
        neighbour_level: int, neighbour_global_id: int,
        edge_code: EdgeCode
    ):
        if edge_code == EDGE_CODE_INVALID:
            return
        
        grid_idx = grid_cache.map[(grid_level, grid_global_id)]
        neighbour_idx = grid_cache.map[(neighbour_level, neighbour_global_id)]
        oppo_code = self._get_toggle_edge_code(edge_code)
        grid_cache.neighbours[grid_idx][edge_code].add(neighbour_idx)
        grid_cache.neighbours[neighbour_idx][oppo_code].add(grid_idx)
    
    def _find_neighbours_along_edge(
        self, grid_cache: GridCache,
        grid_level: int, grid_global_id: int,
        neighbour_level: int, neighbour_global_id: int,
        edge_code: EdgeCode, adjacent_check_func: Callable
    ):
        # Check if neighbour grid is activated (whether if this grid is a leaf node)
        if grid_cache.map.get((neighbour_level, neighbour_global_id)) is not None:
            self._update_grid_neighbour(grid_cache, grid_level, grid_global_id, neighbour_level, neighbour_global_id, edge_code)
        else:
            adj_children: list[tuple[int, int]] = []
            grid_stack: list[tuple[int, int]] = [(neighbour_level, neighbour_global_id)]
            
            while grid_stack:
                _level, _global_id = grid_stack.pop()
                if _level >= len(self.subdivide_rules):
                    continue
                
                sub_width, sub_height = self.subdivide_rules[_level]
                children_global_ids = _get_children_global_ids(_level, _global_id, self.meta_level_info, self.subdivide_rules)
                if children_global_ids is None:
                    continue
                
                for child_local_id, child_global_id in enumerate(children_global_ids):
                    is_adjacent = adjacent_check_func(child_local_id, sub_width, sub_height)
                    if not is_adjacent:
                        continue
                    
                    child_level = _level + 1
                    if grid_cache.has_grid(child_level, child_global_id):
                        adj_children.append((child_level, child_global_id))
                    else:
                        grid_stack.append((child_level, child_global_id))
            
            for child_level, child_global_id in adj_children:
                self._update_grid_neighbour(grid_level, grid_global_id, child_level, child_global_id, edge_code)
                
    def _find_grid_neighbours(self, grid_cache: GridCache):
        for level, global_id in grid_cache.array:
            width = self.meta_level_info[level]['width']
            height = self.meta_level_info[level]['height']
            
            global_u = global_id % width
            global_v = global_id // width
            
            # Check top edge with tGrid
            t_grid = self._get_grid_from_uv(level, width, height, global_u, global_v + 1)
            if t_grid:
                self._find_neighbours_along_edge(grid_cache, level, global_id, t_grid[0], t_grid[1], EdgeCode.NORTH, ADJACENT_CHECK_NORTH)
            # Check left edge with lGrid
            l_grid = self._get_grid_from_uv(level, width, height, global_u - 1, global_v)
            if l_grid:
                self._find_neighbours_along_edge(grid_cache, level, global_id, l_grid[0], l_grid[1], EdgeCode.WEST, ADJACENT_CHECK_WEST)
            # Check bottom edge with bGrid
            b_grid = self._get_grid_from_uv(level, width, height, global_u, global_v - 1)
            if b_grid:
                self._find_neighbours_along_edge(grid_cache, level, global_id, b_grid[0], b_grid[1], EdgeCode.SOUTH, ADJACENT_CHECK_SOUTH)
            # Check right edge with rGrid
            r_grid = self._get_grid_from_uv(level, width, height, global_u + 1, global_v)
            if r_grid:
                self._find_neighbours_along_edge(grid_cache, level, global_id, r_grid[0], r_grid[1], EdgeCode.EAST, ADJACENT_CHECK_EAST)
    
    def _get_fractional_coords(self, level: int, global_id: int) -> tuple[list[int], list[int], list[int], list[int]]:
        width = self.meta_level_info[level]['width']
        height = self.meta_level_info[level]['height']
        
        u = global_id % width
        v = global_id // width
        
        x_min_frac = _simplify_fraction(u, width)
        x_max_frac = _simplify_fraction(u + 1, width)
        y_min_frac = _simplify_fraction(v, height)
        y_max_frac = _simplify_fraction(v + 1, height)
        
        return x_min_frac, x_max_frac, y_min_frac, y_max_frac

    def _get_edge_index(self, grid_index_a: int, grid_index_b: int | None, direction: int, edge_range_info: list[list[int]], code_from_a: EdgeCode) -> bytes:
        if direction not in (0, 1):
            raise ValueError('Direction must be either 0 (vertical) or 1 (horizontal)')
        if not isinstance(edge_range_info, list) or len(edge_range_info) != 3:
            raise ValueError('edge_range_info must be a list of three [numerator, denominator] pairs')
        
        # Unpack the range components. Each is expected to be a UINT32
        min_num, min_den = edge_range_info[0]
        max_num, max_den = edge_range_info[1]
        shared_num, shared_den = edge_range_info[2]
        
        # Ensure canonical ordering for the varying range (min <= max)
        if float(min_num) / float(min_den) > float(max_num) / float(max_den):
            min_num, max_num = max_num, min_num
            min_den, max_den = max_den, min_den
        
        # Construct the edge key (25 bytes total, !BIIIIII)
        # Bit allocation:
        # aligned: 7 bit (highest)
        # direction: 1 bit
        # min_num: 32 bits
        # min_den: 32 bits
        # max_num: 32 bits
        # max_den: 32 bits
        # shared_num: 32 bits
        # shared_den: 32 bits
        # Total bits = 1 + 7 + 32 * 6 = 200 bits (25 bytes)
        edge_key = struct.pack(
            '!BIIIIII',
            1 if direction else 0,
            min_num, min_den,
            max_num, max_den,
            shared_num, shared_den
        )
        
        # Try get edge_index
        if edge_key not in self._edge_index_dict:
            edge_index = len(self._edge_index_cache)
            self._edge_index_dict[edge_key] = edge_index
            self._edge_index_cache.append(edge_key)

            grids = [grid_index_b, grid_index_a] if code_from_a == EdgeCode.NORTH or code_from_a == EdgeCode.WEST else [grid_index_a, grid_index_b]
            self._edge_adj_grids_indices.append(grids)
            return edge_index
        else:
            return self._edge_index_dict[edge_key]
    
    def _add_grid_edge(
        self,
        grid_cache: GridCache, grid_index: int,
        edge_code: EdgeCode, edge_index: int
    ):
        grid_cache.edges[grid_index][edge_code].add(edge_index)
    
    def _calc_horizontal_edges(
        self, grid_cache: GridCache,
        grid_index: int, level: int,
        neighbour_indices: list[int],
        edge_code: EdgeCode, op_edge_code: EdgeCode,
        shared_y_frac: list[int]
    ):
        grid_x_min_frac, grid_x_max_frac, _, _ = grid_cache.fract_coords[grid_index]
        grid_x_min, grid_x_max = grid_x_min_frac[0] / grid_x_min_frac[1], grid_x_max_frac[0] / grid_x_max_frac[1]
        
        # Case when no neighbour ##################################################
        if not neighbour_indices:
            edge_index = self._get_edge_index(grid_index, None, 1, [grid_x_min_frac, grid_x_max_frac, shared_y_frac], edge_code)
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            return
        
        # Case when neighbour has lower level ##################################################
        if len(neighbour_indices) == 1 and grid_cache.array[neighbour_indices[0]][0] < level:
            edge_index = self._get_edge_index(grid_index, neighbour_indices[0], 1, [grid_x_min_frac, grid_x_max_frac, shared_y_frac], edge_code)
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            self._add_grid_edge(grid_cache, neighbour_indices[0], op_edge_code, edge_index)
            return
        
        # Case when neighbours have equal or higher levels ##################################################
        processed_neighbours = []
        for n_grid_index in neighbour_indices:
            n_x_min_frac, n_x_max_frac, _, _ = grid_cache.fract_coords[n_grid_index]
            processed_neighbours.append({
                'index': n_grid_index,
                'x_min_frac': n_x_min_frac,
                'x_max_frac': n_x_max_frac,
                'x_min': n_x_min_frac[0] / n_x_min_frac[1],
                'x_max': n_x_max_frac[0] / n_x_max_frac[1],
            })
            
        # Sort neighbours by their x_min
        processed_neighbours.sort(key=lambda n: n['x_min'])

        # Calculate edge between grid xMin and first neighbour if existed
        if grid_x_min != processed_neighbours[0]['x_min']:
            edge_index = self._get_edge_index(
                grid_index, None, 1,
                [grid_x_min_frac, processed_neighbours[0]['x_min_frac'], shared_y_frac], edge_code
            )
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        
        # Calculate edges between neighbours
        for i in range(len(processed_neighbours) - 1):
            neighbour_from = processed_neighbours[i]
            neighbour_to = processed_neighbours[i + 1]
            
            # Calculate edge of neighbour_from
            edge_index = self._get_edge_index(
                grid_index, neighbour_from['index'], 1,
                [neighbour_from['x_min_frac'], neighbour_from['x_max_frac'], shared_y_frac], edge_code
            )
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            self._add_grid_edge(grid_cache, neighbour_from['index'], op_edge_code, edge_index)
            
            # Calculate edge between neighbourFrom and neighbourTo if existed
            if neighbour_from['x_max'] != neighbour_to['x_min']:
                edge_index = self._get_edge_index(
                    grid_index, None, 1,
                    [neighbour_from['x_max_frac'], neighbour_to['x_min_frac'], shared_y_frac], edge_code
                )
                self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
                
        # Calculate edge of last neighbour
        neighbour_last = processed_neighbours[-1]
        edge_index = self._get_edge_index(
            grid_index, neighbour_last['index'], 1,
            [neighbour_last['x_min_frac'], neighbour_last['x_max_frac'], shared_y_frac], edge_code
        )
        self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        self._add_grid_edge(grid_cache, neighbour_last['index'], op_edge_code, edge_index)
        
        # Calculate edge between last neighbour and grid xMax if existed
        if grid_x_max != neighbour_last['x_max']:
            edge_index = self._get_edge_index(
                grid_index, None, 1,
                [neighbour_last['x_max_frac'], grid_x_max_frac, shared_y_frac], edge_code
            )
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
    
    def _calc_vertical_edges(
        self, grid_cache: GridCache,
        grid_index: int, level: int,
        neighbour_indices: list[int],
        edge_code: EdgeCode, op_edge_code: EdgeCode,
        shared_x_frac: list[int]
    ):
        _, _, grid_y_min_frac, grid_y_max_frac = grid_cache.fract_coords[grid_index]
        grid_y_min, grid_y_max = grid_y_min_frac[0] / grid_y_min_frac[1], grid_y_max_frac[0] / grid_y_max_frac[1]
        
        # Case when no neighbour ##################################################
        if not neighbour_indices:
            edge_index = self._get_edge_index(grid_index, None, 0, [grid_y_min_frac, grid_y_max_frac, shared_x_frac], edge_code)
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            return
        
        # Case when neighbour has lower level ##################################################
        if len(neighbour_indices) == 1 and grid_cache.array[neighbour_indices[0]][0] < level:
            edge_index = self._get_edge_index(grid_index, neighbour_indices[0], 0, [grid_y_min_frac, grid_y_max_frac, shared_x_frac], edge_code)
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            self._add_grid_edge(grid_cache, neighbour_indices[0], op_edge_code, edge_index)
            return
        
        # Case when neighbours have equal or higher levels ##################################################
        processed_neighbours = []
        for n_grid_index in neighbour_indices:
            _, _, n_y_min_frac, n_y_max_frac = grid_cache.fract_coords[n_grid_index]
            processed_neighbours.append({
                'index': n_grid_index,
                'y_min_frac': n_y_min_frac,
                'y_max_frac': n_y_max_frac,
                'y_min': n_y_min_frac[0] / n_y_min_frac[1],
                'y_max': n_y_max_frac[0] / n_y_max_frac[1],
            })

        # Sort neighbours by their y_min
        processed_neighbours.sort(key=lambda n: n['y_min'])

        # Calculate edge between grid yMin and first neighbour if existed
        if grid_y_min != processed_neighbours[0]['y_min']:
            edge_index = self._get_edge_index(
                grid_index, None, 0,
                [grid_y_min_frac, processed_neighbours[0]['y_min_frac'], shared_x_frac], edge_code
            )
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        
        # Calculate edges between neighbours
        for i in range(len(processed_neighbours) - 1):
            neighbour_from = processed_neighbours[i]
            neighbour_to = processed_neighbours[i + 1]
            
            # Calculate edge of neighbour_from
            edge_index = self._get_edge_index(
                grid_index, neighbour_from['index'], 0,
                [neighbour_from['y_min_frac'], neighbour_from['y_max_frac'], shared_x_frac], edge_code
            )
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            self._add_grid_edge(grid_cache, neighbour_from['index'], op_edge_code, edge_index)
            
            # Calculate edge between neighbourFrom and neighbourTo if existed
            if neighbour_from['y_max'] != neighbour_to['y_min']:
                edge_index = self._get_edge_index(
                    grid_index, None, 0,
                    [neighbour_from['y_max_frac'], neighbour_to['y_min_frac'], shared_x_frac], edge_code
                )
                self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
                
        # Calculate edge of last neighbour
        neighbour_last = processed_neighbours[-1]
        edge_index = self._get_edge_index(
            grid_index, neighbour_last['index'], 0,
            [neighbour_last['y_min_frac'], neighbour_last['y_max_frac'], shared_x_frac], edge_code
        )
        self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        self._add_grid_edge(grid_cache, neighbour_last['index'], op_edge_code, edge_index)

        # Calculate edge between last neighbour and grid yMax if existed
        if grid_y_max != neighbour_last['y_max']:
            edge_index = self._get_edge_index(
                grid_index, None, 0,
                [neighbour_last['y_max_frac'], grid_y_max_frac, shared_x_frac], edge_code
            )
            self._add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            
    def _calc_grid_edges(self, grid_cache: GridCache):
        # Pre-calculate fractional coordinates for each grid
        for level, global_id in grid_cache.array:
            grid_cache.fract_coords.append(self._get_fractional_coords(level, global_id))

        for grid_index, (level, global_id) in enumerate(grid_cache.array):
            neighbours = grid_cache.neighbours[grid_index]
            grid_x_min_frac, grid_x_max_frac, grid_y_min_frac, grid_y_max_frac = grid_cache.fract_coords[grid_index]
            
            north_neighbours = list(neighbours[EdgeCode.NORTH])
            self._calc_horizontal_edges(grid_cache, grid_index, level, north_neighbours, EdgeCode.NORTH, EdgeCode.SOUTH, grid_y_max_frac)
            
            west_neighbours = list(neighbours[EdgeCode.WEST])
            self._calc_vertical_edges(grid_cache, grid_index, level, west_neighbours, EdgeCode.WEST, EdgeCode.EAST, grid_x_min_frac)
            
            south_neighbours = list(neighbours[EdgeCode.SOUTH])
            self._calc_horizontal_edges(grid_cache, grid_index, level, south_neighbours, EdgeCode.SOUTH, EdgeCode.NORTH, grid_y_min_frac)
            
            east_neighbours = list(neighbours[EdgeCode.EAST])
            self._calc_vertical_edges(grid_cache, grid_index, level, east_neighbours, EdgeCode.EAST, EdgeCode.WEST, grid_x_max_frac)

    def _parse_topology(self, grid_cache: GridCache):
        # Step 1: Calculate all grid neighbours
        current_time = time.time()
        self._find_grid_neighbours(grid_cache)
        print(f'Grid neighbour calculation took {time.time() - current_time:.2f} seconds')

        # Step 2: Calculate all grid edges
        current_time = time.time()
        self._calc_grid_edges(grid_cache)
        print(f'Grid edge calculation took {time.time() - current_time:.2f} seconds')
        print(f'Find grid edges: {len(self._edge_index_cache)} edges')
    
    def _create_grid_records(self, grid_cache: GridCache):
        batch_size = 10000
        batch_args = [
            (grid_cache.slice_grids(i, batch_size), grid_cache.slice_edges(i, batch_size), i)
            for i in range(0, len(grid_cache), batch_size)
        ]
        batch_func = partial(
            _batch_grid_records_worker,
            bbox=self.bounds,
            meta_level_info=self.meta_level_info,
            grid_info=self.grid_info
        )
        
        num_processes = min(os.cpu_count(), len(batch_args))
        with mp.Pool(processes=num_processes) as pool:
            grid_records_list = pool.map(batch_func, batch_args)
        grid_records = bytearray()
        for grid_records_chunk in grid_records_list:
            grid_records += grid_records_chunk
        
        with open(self.grid_record_path, 'wb') as f:
            f.write(grid_records)
    
    def _slice_edge_info(self, start_index: int, length: int) -> tuple[list[bytes], list[list[int | None]]]:
        if start_index < 0 or start_index >= len(self._edge_index_cache):
            raise IndexError('Start index out of range')
        end_index = min(start_index + length, len(self._edge_index_cache))
        edge_indices = self._edge_index_cache[start_index:end_index]
        edge_adj_grids_indices = self._edge_adj_grids_indices[start_index:end_index]
        return edge_indices, edge_adj_grids_indices
    
    def _create_edge_records(self):
        batch_size = 10000
        batch_args = [
            (*self._slice_edge_info(i, batch_size), i)
            for i in range(0, len(self._edge_index_cache), batch_size)
        ]
        batch_func = partial(
            _batch_edge_records_worker,
            bbox=self.bounds
        )
        num_processes = min(os.cpu_count(), len(batch_args))
        with mp.Pool(processes=num_processes) as pool:
            edge_records_list = pool.map(batch_func, batch_args)
        edge_records = bytearray()
        for edge_records_chunk in edge_records_list:
            edge_records += edge_records_chunk
        
        with open(self.edge_record_path, 'wb') as f:
            f.write(edge_records)

    def merge(self):
        if self.grid_record_path.exists() and self.edge_record_path.exists():
            return
        
        # Get treeger
        treeger = Treeger()
        
        # Create meta overview
        self._create_meta_overview(treeger)

        # Iterate all patches to process the meta overview
        for patch_info in self.patch_infos:
            self._process_patch(treeger, patch_info)

        # Find all active grids
        active_grid_info = self._find_all_active_grids()
        grid_cache = GridCache(active_grid_info)
        
        # Parse topology information
        self._parse_topology(grid_cache)
        
        # Create records
        self._create_grid_records(grid_cache)
        self._create_edge_records()
        
        # Test
        cursor = 0
        with open(self.grid_record_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for _ in range(10):
                    print('-------------------------------')
                    # Read the grid record bytes by cursor
                    mm.seek(cursor)
                    length_prefix = struct.unpack('!I', mm.read(4))[0]
                    mm.seek(cursor + 4)
                    data = mm.read(length_prefix)
                    cursor += 4 + length_prefix

                    e = HydroElement(data)
                    ne = e.ne
                    
                    index = ne[0]
                    left_edge_num = ne[1]
                    right_edge_num = ne[2]
                    bottom_edge_num = ne[3]
                    center = ne[-4:-1]
                    type = ne[-1]
                    
                    left_edge_start = 0
                    right_edge_start = left_edge_start + left_edge_num
                    bottom_edge_start = right_edge_start + right_edge_num
                    top_edge_start = bottom_edge_start + bottom_edge_num
                    
                    edges = ne[5:-4]
                    left_edges = edges[left_edge_start:right_edge_start]
                    right_edges = edges[right_edge_start:bottom_edge_start]
                    bottom_edges = edges[bottom_edge_start:top_edge_start]
                    top_edges = edges[top_edge_start:]
                    
                    print(f'Element info: Index: {index}, Type: {type}, Center: {center}, Left Edges: {left_edges}, Right Edges: {right_edges}, Bottom Edges: {bottom_edges}, Top Edges: {top_edges}')
        
        print('\n##########')
        print('##########')
        print('##########\n')
        
        cursor = 0
        with open(self.edge_record_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for _ in range(10):
                    print('-------------------------------')
                    # Read the edge record bytes by cursor
                    mm.seek(cursor)
                    length_prefix = struct.unpack('!I', mm.read(4))[0]
                    mm.seek(cursor + 4)
                    data = mm.read(length_prefix)
                    cursor += 4 + length_prefix
                    
                    side = HydroSide(data)
                    ns = side.ns
                    print(f'Edge info: Index: {ns[0]}, direction: {ns[1]}, Grids: {ns[2:6]}, Length: {ns[6]}, Center: {ns[7:10]}, Type: {ns[10]}')

    def parse_grid_records(self) -> list:
        with open(self.grid_record_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                cursor = 0
                grid_list = []
                while cursor < mm.size():
                    mm.seek(cursor)
                    length_prefix = struct.unpack('!I', mm.read(4))[0]
                    mm.seek(cursor + 4)
                    data = mm.read(length_prefix)
                    cursor += 4 + length_prefix
                    
                    e = HydroElement(data)
                    grid_list.append(e)

                return grid_list

    def parse_edge_records(self) -> list:
        with open(self.edge_record_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                cursor = 0
                edge_list = []
                while cursor < mm.size():
                    mm.seek(cursor)
                    length_prefix = struct.unpack('!I', mm.read(4))[0]
                    mm.seek(cursor + 4)
                    data = mm.read(length_prefix)
                    cursor += 4 + length_prefix

                    s = HydroSide(data)
                    edge_list.append(s)
                return edge_list

# Helpers ##################################################

def _encode_index(level: int, global_id: int) -> np.uint64:
    """Encode level and global_id into a single index key"""
    return np.uint64(level) << 32 | np.uint64(global_id)

def _decode_index(encoded: np.uint64) -> tuple[int, int]:
    """Decode the index key into level and global_id"""
    level = int(encoded >> 32)
    global_id = int(encoded & 0xFFFFFFFF)
    return level, global_id

def _cleanup_worker_resources():
    """
    This function will be registered to run when the worker process exits.
    It cleans up the global file handle and mmap object.
    """
    global WORKER_MMAP_OBJ, WORKER_FILE_HANDLE
    if WORKER_MMAP_OBJ:
        WORKER_MMAP_OBJ.close()
        WORKER_MMAP_OBJ = None
    if WORKER_FILE_HANDLE:
        WORKER_FILE_HANDLE.close()
        WORKER_FILE_HANDLE = None

def _init_chunk_overview_worker(patch_instance: Patch, grid_ov_path: str):
    global WORKER_PATCH_OBJ, WORKER_MMAP_OBJ, WORKER_FILE_HANDLE
    WORKER_PATCH_OBJ = patch_instance
    WORKER_FILE_HANDLE = open(grid_ov_path, 'r+b')
    WORKER_MMAP_OBJ = mmap.mmap(WORKER_FILE_HANDLE.fileno(), 0, access=mmap.ACCESS_WRITE)

    atexit.register(_cleanup_worker_resources)

def _process_chunk_overview_worker(
        row_index: int,
        patch_width: int,
        ov_offset: list[int],
        ov_bit_length: int,
        bounds: list[float],
        first_level_grid_size: list[float],
        first_level_width: int,
        ov_byte_length: int,
        meta_level_info: list[dict[str, int]],
        grid_info: list[list[float]]
    ) -> None:
    """
    Worker function to process a single patch overview.
    """
    global WORKER_PATCH_OBJ, WORKER_MMAP_OBJ
    
    mm = WORKER_MMAP_OBJ
    patch = WORKER_PATCH_OBJ
    ov = Overview(ov_bit_length)
    empty_data = bytearray(ov_byte_length)
    row_results_buffer = bytearray(patch_width * ov_byte_length)
    for col_index in range(patch_width):
        ov.data = empty_data[:] # reset overview data for each column
        
        first_level_global_id = row_index * patch_width + col_index
        p_stack = [_encode_index(1, first_level_global_id)]
        while p_stack:
            index = p_stack.pop()
            status = patch.get_status(index)
            level, global_id = _decode_index(index)
            
            # Handle active status
            if status == 0b10:
                offset = ov_offset[level - 1]
                local_id = 0 if level == 1 else _get_meta_local_id(level, global_id, first_level_global_id, meta_level_info, grid_info)
                ov.set_value(offset + local_id, True)

            # Handle inactive (inactive and not deleted) status
            elif status == 0b00:
                children_info = patch.get_children_global_ids(level, global_id)
                if children_info is not None:
                    for child_global_id in children_info:
                        p_stack.append(_encode_index(level + 1, child_global_id))
            # TODO: How to flag 0b01 (deleted) status?
    
        buffer_start = col_index * ov_byte_length
        buffer_end = buffer_start + ov_byte_length
        row_results_buffer[buffer_start:buffer_end] = ov.data
    
    schema = patch.get_schema()
    patch_offset_x = int((schema.bounds[0] - bounds[0]) / first_level_grid_size[0])
    patch_offset_y = int((schema.bounds[1] - bounds[1]) / first_level_grid_size[1])
    row_start_offset_in_file = ((patch_offset_y + row_index) * first_level_width + patch_offset_x) * ov_byte_length
    
    # Get current and meta overview chunks
    write_chunk_size = patch_width * ov_byte_length
    patch_ov_chunk = np.frombuffer(row_results_buffer, dtype=np.uint8)
    meta_ov_chunk = np.frombuffer(mm, dtype=np.uint8, count=write_chunk_size, offset=row_start_offset_in_file)
    
    # Perform bitwise OR operation to merge the patch overview into the meta overview
    np.bitwise_or(meta_ov_chunk, patch_ov_chunk, out=meta_ov_chunk)

    del meta_ov_chunk
    del patch_ov_chunk
    return

def _get_children_global_ids(
        level: int,
        global_id: int,
        meta_level_info: list[dict[str, int]],
        subdivide_rules: list[list[int]]
    ) -> list[int]:
    if (level < 0) or (level >= len(meta_level_info)):
        return []

    width = meta_level_info[level]['width']
    global_u = global_id % width
    global_v = global_id // width
    sub_width = subdivide_rules[level][0]
    sub_height = subdivide_rules[level][1]
    sub_count = sub_width * sub_height
    
    baseGlobalWidth = width * sub_width
    child_global_ids = [0] * sub_count
    for local_id in range(sub_count):
        local_u = local_id % sub_width
        local_v = local_id // sub_width
        
        sub_global_u = global_u * sub_width + local_u
        sub_global_v = global_v * sub_height + local_v
        child_global_ids[local_id] = sub_global_v * baseGlobalWidth + sub_global_u
    
    return child_global_ids

def _encode_grid_to_bytes(level: int, global_id: int) -> bytes:
    """Encode level (uint8) and global_id (uint64) into bytes"""
    return struct.pack('!BQ', level, global_id)

def _get_meta_local_id(
        level: int, global_id: int,
        first_level_global_id: int,
        meta_level_info: list[dict[str, int]],  grid_info: list[list[float]]
) -> int:
    if level == 0 or level == 1:
        return global_id

    sub_width_from_first_level = int(grid_info[0][0] / grid_info[level - 1][0])
    sub_height_from_first_level = int(grid_info[0][1] / grid_info[level - 1][1])

    first_level_width = meta_level_info[1]['width']
    first_level_u = first_level_global_id % first_level_width
    first_level_v = first_level_global_id // first_level_width

    first_level_bl_u = first_level_u * sub_width_from_first_level   # first level bl column num at level: {level}
    first_level_bl_v = first_level_v * sub_height_from_first_level  # first level bl row num at level: {level}

    total_width = meta_level_info[level]['width']
    global_u = global_id % total_width
    global_v = global_id // total_width

    local_u = global_u - first_level_bl_u
    local_v = global_v - first_level_bl_v
    return local_v * sub_width_from_first_level + local_u
    
def _process_overview(
        ov: Overview,
        first_level_global_id: int,
        ov_offset: list[int],
        meta_level_info: list[dict[str, int]],
        subdivide_rules: list[list[int]],
        grid_info: list[list[float]]
    ) -> bytearray:
    active_grid_info: bytearray = bytearray()
    g_stack = [(1, first_level_global_id)] # stack of (level, global_id)
    while g_stack:
        level, global_id = g_stack.pop()
        if level == 1:
            local_id = 0
        else:
            local_id = _get_meta_local_id(level, global_id, first_level_global_id, meta_level_info, grid_info)
        ov_index = ov_offset[level - 1] + local_id

        if ov.get_value(ov_index):
            active_grid_info += _encode_grid_to_bytes(level, global_id)
        else:
            if level >= len(meta_level_info) - 1: # meta_level_info[0] is the root level (not the first), valid length of meta_level_info is len(meta_level_info) - 1
                continue

            children_info = _get_children_global_ids(level, global_id, meta_level_info, subdivide_rules)
            if children_info:
                for child_global_id in children_info:
                    g_stack.append((level + 1, child_global_id))

    return active_grid_info

def _init_batch_process_overview_worker(grid_ov_path: str):
    global WORKER_MMAP_OBJ, WORKER_FILE_HANDLE
    WORKER_FILE_HANDLE = open(grid_ov_path, 'r+b')
    WORKER_MMAP_OBJ = mmap.mmap(WORKER_FILE_HANDLE.fileno(), 0, access=mmap.ACCESS_READ)
    
    atexit.register(_cleanup_worker_resources)

def _batch_process_overview_worker(
        batch_byte_offset: int,
        batch_size: int,
        ov_byte_length: int,
        ov_offset: list[int],
        meta_level_info: list[dict[str, int]],
        subdivide_rules: list[list[int]],
        grid_info: list[list[float]]
    ) -> bytearray:
    global WORKER_MMAP_OBJ
    mm = WORKER_MMAP_OBJ
    
    end = min(batch_byte_offset + batch_size, mm.size())
    batch = mm[batch_byte_offset:end]
    active_grid_info: bytearray = bytearray()
    for i in range(0, len(batch), ov_byte_length):
        ov_bytes_i = batch[i:i + ov_byte_length]
        ov = Overview.create(ov_bytes_i)
        
        # Skip empty overview
        if int.from_bytes(ov.data, byteorder='big', signed=False) == 0:
            continue
        
        active_grid_info += _process_overview(
            ov,
            (batch_byte_offset + i) // ov_byte_length,
            ov_offset,
            meta_level_info,
            subdivide_rules,
            grid_info
        )
    return active_grid_info

def _simplify_fraction(n: int, m: int) -> list[int]:
    """Find the greatest common divisor of two numbers"""
    a, b = n, m
    while b != 0:
        a, b = b, a % b
    return [n // a, m // a]

def _get_grid_coordinates(level: int, global_id: int, bbox: list[float], meta_level_info: list[dict[str, int]], grid_info: list[list[float]]) -> tuple[float, float, float, float]:
    width = meta_level_info[level]['width']
    
    u = global_id % width
    v = global_id // width
    grid_width, grid_height = grid_info[level-1]
    
    min_xs = bbox[0] + u * grid_width
    min_ys = bbox[1] + v * grid_height
    max_xs = min_xs + grid_width
    max_ys = min_ys + grid_height
    return min_xs, min_ys, max_xs, max_ys

def _generate_grid_record(
    index: int,
    key: bytes, edges: list[set[int]], bbox: list[float],
    meta_level_info: list[dict[str, int]], grid_info: list[list[float]]
) -> bytearray:
    level, global_id = struct.unpack('>BQ', key)
    min_xs, min_ys, max_xs, max_ys = _get_grid_coordinates(level, global_id, bbox, meta_level_info, grid_info)

    unpacked_info = [
        index + 1,                                                      # index (1-based)
        min_xs, min_ys, max_xs, max_ys,                                 # grid coordinates
        len(edges[EdgeCode.WEST]),                                      # west edge count
        len(edges[EdgeCode.EAST]),                                      # east edge count
        len(edges[EdgeCode.SOUTH]),                                     # south edge count
        len(edges[EdgeCode.NORTH]),                                     # north edge count
        *[edge_index + 1 for edge_index in edges[EdgeCode.WEST]],       # west edge indices (1-based)
        *[edge_index + 1 for edge_index in edges[EdgeCode.EAST]],       # east edge indices (1-based)
        *[edge_index + 1 for edge_index in edges[EdgeCode.SOUTH]],      # south edge indices (1-based)
        *[edge_index + 1 for edge_index in edges[EdgeCode.NORTH]],      # north edge indices (1-based)
    ]
    
    unpacked_info_type = [
        'Q',                                    # index (uint64)
        'd', 'd', 'd', 'd',                     # grid coordinates (double)
        'B',                                    # west edge count (uint8)
        'B',                                    # east edge count (uint8)
        'B',                                    # south edge count (uint8)
        'B',                                    # north edge count (uint8)
        *['Q'] * len(edges[EdgeCode.WEST]),     # west edge indices (list of uint64)
        *['Q'] * len(edges[EdgeCode.EAST]),     # east edge indices (list of uint64)
        *['Q'] * len(edges[EdgeCode.SOUTH]),    # south edge indices (list of uint64)
        *['Q'] * len(edges[EdgeCode.NORTH]),    # north edge indices (list of uint64)
    ]
    
    packed_record = bytearray()
    for value, value_type in zip(unpacked_info, unpacked_info_type):
        if value_type == 'Q':  # uint64
            packed_record.extend(struct.pack('!Q', value))
        elif value_type == 'B':  # uint8
            packed_record.extend(struct.pack('!B', value))
        elif value_type == 'd':  # double
            packed_record.extend(struct.pack('!d', value))

    return packed_record

def _batch_grid_records_worker(
    args: tuple[bytes, list[list[set[int]]]], bbox: list[float],
    meta_level_info: list[dict[str, int]], grid_info: list[list[float]]
) -> bytearray:
    grid_data, grid_edges, offset = args

    records = bytearray()
    grid_count = len(grid_data) // 9 # each grid has 9 bytes (level: uint8 + global_id: uint64)
    for i in range(grid_count):
        start = i * 9
        end = start + 9
        key = grid_data[start:end]
        
        # Get edges for this grid
        edges = grid_edges[i]
        
        # Generate grid record
        record =  _generate_grid_record(offset + i, key, edges, bbox, meta_level_info, grid_info)
        length_prefix = struct.pack('!I', len(record)) 
        
        records += length_prefix
        records += record

    return records

def _generate_edge_record(index: int, edge_data: bytes, edge_grids: list[int | None], bbox: list[float]) -> bytearray:
    direction, min_num, min_den, max_num, max_den, shared_num, shared_den = struct.unpack('!BIIIIII', edge_data)
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    if direction == 0:  # vertical edge
        x_min = bbox[0] + (shared_num / shared_den) * (bbox[2] - bbox[0])
        x_max = x_min
        y_min = bbox[1] + (min_num / min_den) * (bbox[3] - bbox[1])
        y_max = bbox[1] + (max_num / max_den) * (bbox[3] - bbox[1])
    elif direction == 1:  # horizontal edge
        x_min = bbox[0] + (min_num / min_den) * (bbox[2] - bbox[0])
        x_max = bbox[0] + (max_num / max_den) * (bbox[2] - bbox[0])
        y_min = bbox[1] + (shared_num / shared_den) * (bbox[3] - bbox[1])
        y_max = y_min
    
    return struct.pack(
        '!QBddddQQ',
        index + 1,  # index (1-based)
        direction,
        x_min, y_min, x_max, y_max,
        edge_grids[0] + 1 if edge_grids[0] is not None else 0, # grid_index_a (1-based)
        edge_grids[1] + 1 if edge_grids[1] is not None else 0  # grid_index_b (1-based)
    )

def _batch_edge_records_worker(args: tuple[list[bytes], list[list[int | None]]], bbox: list[float]) -> bytes:
    edge_data, edge_grids, offset = args

    records = bytearray()
    edge_count = len(edge_data)
    for i in range(edge_count):
        edge = edge_data[i]
        
        record = _generate_edge_record(offset + i, edge, edge_grids[i], bbox)
        length_prefix = struct.pack('!I', len(record))
        
        records += length_prefix
        records += record

    return records