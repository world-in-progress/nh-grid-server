import os
import time
import math
import mmap
import json
import struct
from pathlib import Path
from enum import IntEnum
from typing import Callable
import multiprocessing as mp
from functools import partial

from crms.patch import Patch
from crms.hydro import HydroElement, HydroSide

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

class GridCache:
    class _ArrayView:
        def __init__(self, parent: 'GridCache'):
            self._parent = parent
        
        def __getitem__(self, index: int) -> tuple[int, int]:
            if index < 0 or index >= len(self):
                raise IndexError('Index out of bounds')
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
            raise IndexError('Index out of bounds')
        start = start_index * 9
        end = min(start + length * 9, self._len * 9)
        return self.data[start:end]
    
    def slice_edges(self, start_index: int, length: int) -> bytes:
        if start_index < 0 or start_index > self._len:
            raise IndexError('Index out of bounds')
        end_index = min(start_index + length, self._len)
        return self.edges[start_index : end_index]

def encode_cell_key(level: int, global_id: int) -> bytes:
    return struct.pack('!BQ', level, global_id)

def decode_cell_key(key: bytes) -> tuple[int, int]:
    return struct.unpack('!BQ', key)

def get_bounds(patch_paths: list[str]) -> list[float]:
    inf, neg_inf = float('inf'), float('-inf')
    bounds = [inf, inf, neg_inf, neg_inf]   #min_x, min_y, max_x, max_y
    
    for patch_path in patch_paths:
        path = Path(patch_path)
        patch_meta = json.load(open(path / 'patch.meta.json', 'r', encoding='utf-8'))
        patch_bounds = patch_meta['bounds']
        
        bounds[0] = min(bounds[0], patch_bounds[0])
        bounds[1] = min(bounds[1], patch_bounds[1])
        bounds[2] = max(bounds[2], patch_bounds[2])
        bounds[3] = max(bounds[3], patch_bounds[3])
    return bounds

def get_all_ancestor_keys(key: bytes, level_info: list[dict[str, int]], subdivide_rules: list[list[int]]) -> list[bytes]:
    ancestors: list[bytes] = []
    start_child_level, child_global_id = decode_cell_key(key)
    for parent_level in range(start_child_level - 1, 0, -1):   # skip virtual root level 0
        rule = subdivide_rules[parent_level]
        parent_level_cols = level_info[parent_level]['width']
        child_level_cols = level_info[parent_level + 1]['width']
        
        child_gid_u = child_global_id % child_level_cols
        child_gid_v = child_global_id // child_level_cols
        parent_gid_u = child_gid_u // rule[0]
        parent_gid_v = child_gid_v // rule[1]
        parent_global_id = parent_gid_v * parent_level_cols + parent_gid_u
        
        ancestors.append(encode_cell_key(parent_level, parent_global_id))
        child_global_id = parent_global_id
    return ancestors

def update_cells_by_patch(
    keys: set[bytes],
    schema_file_path: str, patch_workspace: str,
    meta_bounds: list[float], meta_level_info: list[dict[str, int]]
):
    # Create patch crm
    patch = Patch(schema_file_path, patch_workspace)
    
    # Calculate bottom-left fraction in meta grid
    patch_bounds = patch.bounds
    bl_col_meta_f = (patch_bounds[0] - meta_bounds[0]) / (meta_bounds[2] - meta_bounds[0])
    bl_row_meta_f = (patch_bounds[1] - meta_bounds[1]) / (meta_bounds[3] - meta_bounds[1])
    
    # Get active grid infos from patch and update keys
    levels, global_ids = patch.get_active_grid_infos()
    for level, global_id in zip(levels, global_ids):
        # Meta level info
        meta_level_cols = meta_level_info[level]['width']
        meta_level_rows = meta_level_info[level]['height']
        
        # Patch level info
        patch_level_cols = patch.level_info[level]['width']
        
        # Adjust patch global id to meta grid global id
        patch_gid_u = global_id % patch_level_cols
        patch_gid_v = global_id // patch_level_cols

        meta_gid_u = int(bl_col_meta_f * meta_level_cols) + patch_gid_u
        meta_gid_v = int(bl_row_meta_f * meta_level_rows) + patch_gid_v
        meta_global_id = meta_gid_v * meta_level_cols + meta_gid_u
        
        # Encode and add to keys
        cell_key = encode_cell_key(level, meta_global_id)
        keys.add(cell_key)
     
def get_grid_from_uv(level: int, level_width, level_height, u: int, v: int, meta_level_info: list[dict[str, int]]) -> tuple[int, int] | None:
    if level >= len(meta_level_info) or level < 0:
        return None
    
    if u < 0 or u >= level_width or v < 0 or v >= level_height:
        return None
    
    global_id = v * level_width + u
    return (level, global_id)
    
def get_toggle_edge_code(code: int) -> int:
    return TOGGLE_EDGE_CODE_MAP.get(code, EDGE_CODE_INVALID)
    
def update_grid_neighbour(
    grid_cache: GridCache, 
    grid_level: int, grid_global_id: int, 
    neighbour_level: int, neighbour_global_id: int,
    edge_code: EdgeCode
):
    if edge_code == EDGE_CODE_INVALID:
        return
    
    grid_idx = grid_cache.map[(grid_level, grid_global_id)]
    neighbour_idx = grid_cache.map[(neighbour_level, neighbour_global_id)]
    oppo_code = get_toggle_edge_code(edge_code)
    grid_cache.neighbours[grid_idx][edge_code].add(neighbour_idx)
    grid_cache.neighbours[neighbour_idx][oppo_code].add(grid_idx)

def get_children_global_ids(
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

def find_neighbours_along_edge(
    grid_cache: GridCache,
    subdivide_rules: list[list[int]],
    meta_level_info: list[dict[str, int]],
    grid_level: int, grid_global_id: int,
    neighbour_level: int, neighbour_global_id: int,
    edge_code: EdgeCode, adjacent_check_func: Callable
):
    # Check if neighbour grid is activated (whether if this grid is a leaf node)
    if grid_cache.map.get((neighbour_level, neighbour_global_id)) is not None:
        update_grid_neighbour(grid_cache, grid_level, grid_global_id, neighbour_level, neighbour_global_id, edge_code)
    else:
        adj_children: list[tuple[int, int]] = []
        grid_stack: list[tuple[int, int]] = [(neighbour_level, neighbour_global_id)]
        
        while grid_stack:
            _level, _global_id = grid_stack.pop()
            if _level >= len(subdivide_rules):
                continue
            
            sub_width, sub_height = subdivide_rules[_level]
            children_global_ids = get_children_global_ids(_level, _global_id, meta_level_info, subdivide_rules)
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
            update_grid_neighbour(grid_cache, grid_level, grid_global_id, child_level, child_global_id, edge_code)
        

def find_grid_neighbours(grid_cache: GridCache, subdivide_rules: list[list[int]], meta_level_info: list[dict[str, int]]):
    for level, global_id in grid_cache.array:
        width = meta_level_info[level]['width']
        height = meta_level_info[level]['height']
        
        global_u = global_id % width
        global_v = global_id // width
        
        # Check top edge with tGrid
        t_grid = get_grid_from_uv(level, width, height, global_u, global_v + 1, meta_level_info)
        if t_grid:
            find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, t_grid[0], t_grid[1], EdgeCode.NORTH, ADJACENT_CHECK_NORTH)
        # Check left edge with lGrid
        l_grid = get_grid_from_uv(level, width, height, global_u - 1, global_v, meta_level_info)
        if l_grid:
            find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, l_grid[0], l_grid[1], EdgeCode.WEST, ADJACENT_CHECK_WEST)
        # Check bottom edge with bGrid
        b_grid = get_grid_from_uv(level, width, height, global_u, global_v - 1, meta_level_info)
        if b_grid:
            find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, b_grid[0], b_grid[1], EdgeCode.SOUTH, ADJACENT_CHECK_SOUTH)
        # Check right edge with rGrid
        r_grid = get_grid_from_uv(level, width, height, global_u + 1, global_v, meta_level_info)
        if r_grid:
            find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, r_grid[0], r_grid[1], EdgeCode.EAST, ADJACENT_CHECK_EAST)

def simplify_fraction(n: int, m: int) -> list[int]:
    """Find the greatest common divisor of two numbers"""
    a, b = n, m
    while b != 0:
        a, b = b, a % b
    return [n // a, m // a]

def get_fractional_coords(level: int, global_id: int, meta_level_info: list[dict[str, int]]) -> tuple[list[int], list[int], list[int], list[int]]:
    width = meta_level_info[level]['width']
    height = meta_level_info[level]['height']
    
    u = global_id % width
    v = global_id // width
    
    x_min_frac = simplify_fraction(u, width)
    x_max_frac = simplify_fraction(u + 1, width)
    y_min_frac = simplify_fraction(v, height)
    y_max_frac = simplify_fraction(v + 1, height)
    
    return x_min_frac, x_max_frac, y_min_frac, y_max_frac

def get_edge_index(
    grid_index_a: int, grid_index_b: int | None, 
    direction: int, edge_range_info: list[list[int]], code_from_a: EdgeCode,
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_grids_indices: list[list[int | None]]
) -> bytes:
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
    if edge_key not in edge_index_dict:
        edge_index = len(edge_index_cache)
        edge_index_dict[edge_key] = edge_index
        edge_index_cache.append(edge_key)

        grids = [grid_index_b, grid_index_a] if code_from_a == EdgeCode.NORTH or code_from_a == EdgeCode.WEST else [grid_index_a, grid_index_b]
        edge_adj_grids_indices.append(grids)
        return edge_index
    else:
        return edge_index_dict[edge_key]

    
def add_grid_edge(
    grid_cache: GridCache, grid_index: int,
    edge_code: EdgeCode, edge_index: int
):
    grid_cache.edges[grid_index][edge_code].add(edge_index)

def calc_horizontal_edges(
    grid_cache: GridCache,
    grid_index: int, level: int,
    neighbour_indices: list[int],
    edge_code: EdgeCode, op_edge_code: EdgeCode,
    shared_y_frac: list[int],
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_grids_indices: list[list[int | None]]
):
    grid_x_min_frac, grid_x_max_frac, _, _ = grid_cache.fract_coords[grid_index]
    grid_x_min, grid_x_max = grid_x_min_frac[0] / grid_x_min_frac[1], grid_x_max_frac[0] / grid_x_max_frac[1]
    
    # Case when no neighbour ##################################################
    if not neighbour_indices:
        edge_index = get_edge_index(grid_index, None, 1, [grid_x_min_frac, grid_x_max_frac, shared_y_frac], edge_code, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        return
    
    # Case when neighbour has lower level ##################################################
    if len(neighbour_indices) == 1 and grid_cache.array[neighbour_indices[0]][0] < level:
        edge_index = get_edge_index(grid_index, neighbour_indices[0], 1, [grid_x_min_frac, grid_x_max_frac, shared_y_frac], edge_code, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        add_grid_edge(grid_cache, neighbour_indices[0], op_edge_code, edge_index)
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
        edge_index = get_edge_index(
            grid_index, None, 1,
            [grid_x_min_frac, processed_neighbours[0]['x_min_frac'], shared_y_frac], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_grids_indices
        )
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
    
    # Calculate edges between neighbours
    for i in range(len(processed_neighbours) - 1):
        neighbour_from = processed_neighbours[i]
        neighbour_to = processed_neighbours[i + 1]
        
        # Calculate edge of neighbour_from
        edge_index = get_edge_index(
            grid_index, neighbour_from['index'], 1,
            [neighbour_from['x_min_frac'], neighbour_from['x_max_frac'], shared_y_frac], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_grids_indices
        )
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        add_grid_edge(grid_cache, neighbour_from['index'], op_edge_code, edge_index)
        
        # Calculate edge between neighbourFrom and neighbourTo if existed
        if neighbour_from['x_max'] != neighbour_to['x_min']:
            edge_index = get_edge_index(
                grid_index, None, 1,
                [neighbour_from['x_max_frac'], neighbour_to['x_min_frac'], shared_y_frac], edge_code,
                edge_index_cache, edge_index_dict, edge_adj_grids_indices
            )
            add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            
    # Calculate edge of last neighbour
    neighbour_last = processed_neighbours[-1]
    edge_index = get_edge_index(
        grid_index, neighbour_last['index'], 1,
        [neighbour_last['x_min_frac'], neighbour_last['x_max_frac'], shared_y_frac], edge_code,
        edge_index_cache, edge_index_dict, edge_adj_grids_indices
    )
    add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
    add_grid_edge(grid_cache, neighbour_last['index'], op_edge_code, edge_index)
    
    # Calculate edge between last neighbour and grid xMax if existed
    if grid_x_max != neighbour_last['x_max']:
        edge_index = get_edge_index(
            grid_index, None, 1,
            [neighbour_last['x_max_frac'], grid_x_max_frac, shared_y_frac], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_grids_indices
        )
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)

def calc_vertical_edges(
    grid_cache: GridCache,
    grid_index: int, level: int,
    neighbour_indices: list[int],
    edge_code: EdgeCode, op_edge_code: EdgeCode,
    shared_x_frac: list[int],
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_grids_indices: list[list[int | None]]
):
    _, _, grid_y_min_frac, grid_y_max_frac = grid_cache.fract_coords[grid_index]
    grid_y_min, grid_y_max = grid_y_min_frac[0] / grid_y_min_frac[1], grid_y_max_frac[0] / grid_y_max_frac[1]
    
    # Case when no neighbour ##################################################
    if not neighbour_indices:
        edge_index = get_edge_index(grid_index, None, 0, [grid_y_min_frac, grid_y_max_frac, shared_x_frac], edge_code, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        return
    
    # Case when neighbour has lower level ##################################################
    if len(neighbour_indices) == 1 and grid_cache.array[neighbour_indices[0]][0] < level:
        edge_index = get_edge_index(grid_index, neighbour_indices[0], 0, [grid_y_min_frac, grid_y_max_frac, shared_x_frac], edge_code, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        add_grid_edge(grid_cache, neighbour_indices[0], op_edge_code, edge_index)
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
        edge_index = get_edge_index(
            grid_index, None, 0,
            [grid_y_min_frac, processed_neighbours[0]['y_min_frac'], shared_x_frac], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_grids_indices
        )
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
    
    # Calculate edges between neighbours
    for i in range(len(processed_neighbours) - 1):
        neighbour_from = processed_neighbours[i]
        neighbour_to = processed_neighbours[i + 1]
        
        # Calculate edge of neighbour_from
        edge_index = get_edge_index(
            grid_index, neighbour_from['index'], 0,
            [neighbour_from['y_min_frac'], neighbour_from['y_max_frac'], shared_x_frac], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_grids_indices
        )
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
        add_grid_edge(grid_cache, neighbour_from['index'], op_edge_code, edge_index)
        
        # Calculate edge between neighbourFrom and neighbourTo if existed
        if neighbour_from['y_max'] != neighbour_to['y_min']:
            edge_index = get_edge_index(
                grid_index, None, 0,
                [neighbour_from['y_max_frac'], neighbour_to['y_min_frac'], shared_x_frac], edge_code,
                edge_index_cache, edge_index_dict, edge_adj_grids_indices
            )
            add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            
    # Calculate edge of last neighbour
    neighbour_last = processed_neighbours[-1]
    edge_index = get_edge_index(
        grid_index, neighbour_last['index'], 0,
        [neighbour_last['y_min_frac'], neighbour_last['y_max_frac'], shared_x_frac], edge_code,
        edge_index_cache, edge_index_dict, edge_adj_grids_indices
    )
    add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
    add_grid_edge(grid_cache, neighbour_last['index'], op_edge_code, edge_index)

    # Calculate edge between last neighbour and grid yMax if existed
    if grid_y_max != neighbour_last['y_max']:
        edge_index = get_edge_index(
            grid_index, None, 0,
            [neighbour_last['y_max_frac'], grid_y_max_frac, shared_x_frac], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_grids_indices
        )
        add_grid_edge(grid_cache, grid_index, edge_code, edge_index)
            
def calc_grid_edges(
    grid_cache: GridCache,
    meta_level_info: list[dict[str, int]],
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_grids_indices: list[list[int | None]]
):
    # Pre-calculate fractional coordinates for each grid
    for level, global_id in grid_cache.array:
        grid_cache.fract_coords.append(get_fractional_coords(level, global_id, meta_level_info))

    for grid_index, (level, global_id) in enumerate(grid_cache.array):
        neighbours = grid_cache.neighbours[grid_index]
        grid_x_min_frac, grid_x_max_frac, grid_y_min_frac, grid_y_max_frac = grid_cache.fract_coords[grid_index]
        
        north_neighbours = list(neighbours[EdgeCode.NORTH])
        calc_horizontal_edges(grid_cache, grid_index, level, north_neighbours, EdgeCode.NORTH, EdgeCode.SOUTH, grid_y_max_frac, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
        
        west_neighbours = list(neighbours[EdgeCode.WEST])
        calc_vertical_edges(grid_cache, grid_index, level, west_neighbours, EdgeCode.WEST, EdgeCode.EAST, grid_x_min_frac, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
        
        south_neighbours = list(neighbours[EdgeCode.SOUTH])
        calc_horizontal_edges(grid_cache, grid_index, level, south_neighbours, EdgeCode.SOUTH, EdgeCode.NORTH, grid_y_min_frac, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
        
        east_neighbours = list(neighbours[EdgeCode.EAST])
        calc_vertical_edges(grid_cache, grid_index, level, east_neighbours, EdgeCode.EAST, EdgeCode.WEST, grid_x_max_frac, edge_index_cache, edge_index_dict, edge_adj_grids_indices)

def parse_topology(
    grid_cache: GridCache,
    subdivide_rules: list[list[int]],
    meta_level_info: list[dict[str, int]],
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_grids_indices: list[list[int | None]]
):
    # Step 1: Calculate all grid neighbours
    current_time = time.time()
    find_grid_neighbours(grid_cache, subdivide_rules, meta_level_info)
    print(f'Grid neighbour calculation took {time.time() - current_time:.2f} seconds')

    # Step 2: Calculate all grid edges
    current_time = time.time()
    calc_grid_edges(grid_cache, meta_level_info, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
    print(f'Grid edge calculation took {time.time() - current_time:.2f} seconds')
    print(f'Find grid edges: {len(edge_index_cache)} edges')

def get_grid_coordinates(level: int, global_id: int, bbox: list[float], meta_level_info: list[dict[str, int]], grid_info: list[list[float]]) -> tuple[float, float, float, float]:
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
    min_xs, min_ys, max_xs, max_ys = get_grid_coordinates(level, global_id, bbox, meta_level_info, grid_info)

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

def batch_grid_records_worker(
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

def create_grid_records(
    grid_cache: GridCache,
    meta_bounds: list[float],
    meta_level_info: list[dict[str, int]],
    grid_info: list[list[float]],
    grid_record_path: str
):
    batch_size = 10000
    batch_args = [
        (grid_cache.slice_grids(i, batch_size), grid_cache.slice_edges(i, batch_size), i)
        for i in range(0, len(grid_cache), batch_size)
    ]
    batch_func = partial(
        batch_grid_records_worker,
        bbox=meta_bounds,
        meta_level_info=meta_level_info,
        grid_info=grid_info
    )
    
    num_processes = min(os.cpu_count(), len(batch_args))
    with mp.Pool(processes=num_processes) as pool:
        grid_records_list = pool.map(batch_func, batch_args)
    grid_records = bytearray()
    for grid_records_chunk in grid_records_list:
        grid_records += grid_records_chunk
    
    with open(grid_record_path, 'wb') as f:
        f.write(grid_records)

def slice_edge_info(
    start_index: int, length: int,
    edge_index_cache: list[bytes],
    edge_adj_grids_indices: list[list[int | None]]
) -> tuple[list[bytes], list[list[int | None]]]:
    if start_index < 0 or start_index >= len(   edge_index_cache):
        raise IndexError('Start index out of range')
    end_index = min(start_index + length, len(  edge_index_cache))
    edge_indices = edge_index_cache[start_index:end_index]
    edge_adj_grids_indices = edge_adj_grids_indices[start_index:end_index]
    return edge_indices, edge_adj_grids_indices

def generate_edge_record(index: int, edge_data: bytes, edge_grids: list[int | None], bbox: list[float]) -> bytearray:
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

def batch_edge_records_worker(args: tuple[list[bytes], list[list[int | None]]], bbox: list[float]) -> bytes:
    edge_data, edge_grids, offset = args

    records = bytearray()
    edge_count = len(edge_data)
    for i in range(edge_count):
        edge = edge_data[i]
        
        record = generate_edge_record(offset + i, edge, edge_grids[i], bbox)
        length_prefix = struct.pack('!I', len(record))
        
        records += length_prefix
        records += record

    return records

def create_edge_records(
    edge_index_cache: list[bytes],
    edge_adj_grids_indices: list[list[int | None]],
    meta_bounds: list[float],
    edge_record_path: str
):
    batch_size = 10000
    batch_args = [
        (*slice_edge_info(i, batch_size, edge_index_cache, edge_adj_grids_indices), i)
        for i in range(0, len(edge_index_cache), batch_size)
    ]
    batch_func = partial(
        batch_edge_records_worker,
        bbox=meta_bounds
    )
    num_processes = min(os.cpu_count(), len(batch_args))
    with mp.Pool(processes=num_processes) as pool:
        edge_records_list = pool.map(batch_func, batch_args)
    edge_records = bytearray()
    for edge_records_chunk in edge_records_list:
        edge_records += edge_records_chunk
    
    with open(edge_record_path, 'wb') as f:
        f.write(edge_records)

def assembly(schema_file_path: str, patch_paths: list[str], workspace: str):
    # Initialization ##################################################
    
    # Create workspace directory
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Init schema info
    schema_path = Path(schema_file_path)
    schema = json.load(open(schema_path, 'r', encoding='utf-8'))
    epsg: int = schema['epsg']
    grid_info: list[list[float]] = schema['grid_info']
    first_level_resolution: list[float] = grid_info[0]
    
    # Init bounds from all patches
    meta_bounds = get_bounds(patch_paths)
    
    # Init subdivide rules
    subdivide_rules: list[list[int]] = [
        [
            int(math.ceil((meta_bounds[2] - meta_bounds[0]) / first_level_resolution[0])),
            int(math.ceil((meta_bounds[3] - meta_bounds[1]) / first_level_resolution[1]))
        ]
    ]
    for i in range(len(grid_info) - 1):
        from_resolution = grid_info[i]
        to_resolution = grid_info[i + 1]
        subdivide_rules.append([
            int(from_resolution[0] / to_resolution[0]),
            int(from_resolution[1] / to_resolution[1])
        ])
    subdivide_rules.append([1, 1])  # last level (no subdivision)
    
    # Init meta level info and first level cols/rows
    meta_level_info: list[dict[str, int]] = [{'width': 1, 'height': 1}]
    for level, rule in enumerate(subdivide_rules[:-1]):
        from_cols, from_rows = meta_level_info[level]['width'], meta_level_info[level]['height']
        meta_level_info.append({
            'width': from_cols * rule[0],
            'height': from_rows * rule[1]
        })
    
    # Find activated cells in all patches ##################################################
    
    # Set activated cell key container
    # Key: uin8 level + uint64 global id
    activated_cell_keys: set[bytes] = set()
    
    # Update activated cells by each patch
    for patch_path in patch_paths:
        update_cells_by_patch(
            activated_cell_keys,
            schema_file_path, patch_path,
            meta_bounds, meta_level_info
        )
    
    # Filter activated cells to remove conflicts
    # Conflict: if a cell is activated, all its ancestors must be deactivated
    for level in range(len(meta_level_info), 1, -1):    # from highest level to level 2 (level 1 has no parent)
        keys_at_level = [k for k in activated_cell_keys if k[0] == level]
        ancestor_keys_to_remove: set[bytes] = set()
        for key in keys_at_level:
            ancestor_keys = get_all_ancestor_keys(key, meta_level_info, subdivide_rules)
            ancestor_keys_to_remove.update(ancestor_keys)
        # Batch remove ancestor keys from activated cells in the level
        activated_cell_keys.difference_update(ancestor_keys_to_remove)
    
    # Topology construction for the grid ##################################################
    
    # Sort and concatenate activated cell keys
    keys_data = b''.join(sorted(activated_cell_keys))
    
    # Initialize GridCache
    grid_cache = GridCache(keys_data)
    
    # Initialize edge topology containers
    edge_index_cache: list[bytes] = []
    edge_index_dict: dict[int, bytes] = {}
    edge_adj_grids_indices: list[list[int | None]] = [] # for each edge, the list of adjacent grid indices, among [grid_a, grid_b], grid_a must be the north or west grid

    parse_topology(grid_cache, subdivide_rules, meta_level_info, edge_index_cache, edge_index_dict, edge_adj_grids_indices)
    
    # Create grid records
    grid_record_path = workspace / 'grid_records.bin'
    create_grid_records(
        grid_cache,
        meta_bounds,
        meta_level_info,
        grid_info,
        str(grid_record_path)
    )
    
    # Create edge records
    edge_record_path = workspace / 'edge_records.bin'
    create_edge_records(
        edge_index_cache,
        edge_adj_grids_indices,
        meta_bounds,
        str(edge_record_path)
    )
    
    # Test ##################################################
    cursor = 0
    with open(grid_record_path, 'r+b') as f:
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
    with open(edge_record_path, 'r+b') as f:
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
    