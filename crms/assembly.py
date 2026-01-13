import gc
import os
import time
import math
import json
import struct
from pathlib import Path
from enum import IntEnum
from typing import Callable
import multiprocessing as mp
from functools import partial

from crms.patch import Patch

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

    def has_cell(self, level: int, global_id: int) -> bool:
        return (level, global_id) in self.map

    def slice_cells(self, start_index: int, length: int) -> bytes:
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

def _encode_cell_key(level: int, global_id: int) -> bytes:
    return struct.pack('!BQ', level, global_id)

def _decode_cell_key(key: bytes) -> tuple[int, int]:
    return struct.unpack('!BQ', key)

def _get_bounds(patch_paths: list[str]) -> list[float]:
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

def _get_all_ancestor_keys(key: bytes, level_info: list[dict[str, int]], subdivide_rules: list[list[int]]) -> list[bytes]:
    ancestors: list[bytes] = []
    start_child_level, child_global_id = _decode_cell_key(key)
    for parent_level in range(start_child_level - 1, 0, -1):   # skip virtual root level 0
        rule = subdivide_rules[parent_level]
        parent_level_cols = level_info[parent_level]['width']
        child_level_cols = level_info[parent_level + 1]['width']
        
        child_gid_u = child_global_id % child_level_cols
        child_gid_v = child_global_id // child_level_cols
        parent_gid_u = child_gid_u // rule[0]
        parent_gid_v = child_gid_v // rule[1]
        parent_global_id = parent_gid_v * parent_level_cols + parent_gid_u
        
        ancestors.append(_encode_cell_key(parent_level, parent_global_id))
        child_global_id = parent_global_id
    return ancestors

def _update_cells_by_patch(
    keys: set[bytes],
    schema_file_path: str, patch_workspace: str,
    meta_bounds: list[float], meta_level_info: list[dict[str, int]]
):
    print('Updating meta grid cells by patch:', patch_workspace)
    
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

        meta_gid_u = int(bl_col_meta_f * meta_level_cols + 0.5) + patch_gid_u
        meta_gid_v = int(bl_row_meta_f * meta_level_rows + 0.5) + patch_gid_v
        meta_global_id = meta_gid_v * meta_level_cols + meta_gid_u
        
        # Encode and add to keys
        cell_key = _encode_cell_key(level, meta_global_id)
        keys.add(cell_key)
     
def _get_cell_from_uv(level: int, level_cols, level_rows, u: int, v: int, meta_level_info: list[dict[str, int]]) -> tuple[int, int] | None:
    if level >= len(meta_level_info) or level < 0:
        return None
    
    if u < 0 or u >= level_cols or v < 0 or v >= level_rows:
        return None
    
    global_id = v * level_cols + u
    return level, global_id
    
def _get_toggle_edge_code(code: int) -> int:
    return TOGGLE_EDGE_CODE_MAP.get(code, EDGE_CODE_INVALID)
    
def _update_cell_neighbour(
    grid_cache: GridCache, 
    cell_level: int, cell_global_id: int, 
    neighbour_level: int, neighbour_global_id: int,
    edge_code: EdgeCode
):
    if edge_code == EDGE_CODE_INVALID:
        return
    
    grid_idx = grid_cache.map[(cell_level, cell_global_id)]
    neighbour_idx = grid_cache.map[(neighbour_level, neighbour_global_id)]
    oppo_code = _get_toggle_edge_code(edge_code)
    grid_cache.neighbours[grid_idx][edge_code].add(neighbour_idx)
    grid_cache.neighbours[neighbour_idx][oppo_code].add(grid_idx)

def _get_children_global_ids(
        level: int,
        global_id: int,
        meta_level_info: list[dict[str, int]],
        subdivide_rules: list[list[int]]
) -> list[int]:
    if (level < 0) or (level >= len(meta_level_info)):
        return []

    cols = meta_level_info[level]['width']
    global_u = global_id % cols
    global_v = global_id // cols
    sub_width = subdivide_rules[level][0]
    sub_height = subdivide_rules[level][1]
    sub_count = sub_width * sub_height
    
    sub_total_cols = cols * sub_width
    child_global_ids = [0] * sub_count
    for local_id in range(sub_count):
        local_u = local_id % sub_width
        local_v = local_id // sub_width
        
        sub_global_u = global_u * sub_width + local_u
        sub_global_v = global_v * sub_height + local_v
        child_global_ids[local_id] = sub_global_v * sub_total_cols + sub_global_u
    
    return child_global_ids

def _check_risk_along_edge(
    risk_threshold: int,
    cell_keys: set[bytes],
    subdivide_rules: list[list[int]],
    meta_level_info: list[dict[str, int]],
    cell_level: int,
    neighbour_level: int, neighbour_global_id: int,
    adjacent_check_func: Callable
) -> bool:
    """
    Check if the cell is risk along the edge with neighbour cells
    Risk cells are those cells that has lower level than a specific neighbour cell, while the level difference is greater than risk_threshold
    """
    # Check if neighbour cell is activated (whether if this cell is a leaf node)
    neighbour_key = _encode_cell_key(neighbour_level, neighbour_global_id)
    if neighbour_key in cell_keys:
        return False    # not risk because neighbour cell share a same level
    else:
        cell_stack: list[tuple[int, int]] = [(neighbour_level, neighbour_global_id)]
        
        while cell_stack:
            _level, _global_id = cell_stack.pop()
            if _level >= len(subdivide_rules):
                continue
            
            sub_width, sub_height = subdivide_rules[_level]
            children_global_ids = _get_children_global_ids(_level, _global_id, meta_level_info, subdivide_rules)
            if children_global_ids is None:
                continue
            
            for child_local_id, child_global_id in enumerate(children_global_ids):
                is_adjacent = adjacent_check_func(child_local_id, sub_width, sub_height)
                if not is_adjacent:
                    continue
                
                child_level = _level + 1
                child_key = _encode_cell_key(child_level, child_global_id)
                if child_key in cell_keys:
                    if child_level - cell_level > risk_threshold:
                        return True # risk found
                else:
                    cell_stack.append((child_level, child_global_id))
    return False
        
def _find_risk_cells(
    risk_threshold: int, cell_keys: set[bytes],
    subdivide_rules: list[list[int]], meta_level_info: list[dict[str, int]]
) -> set[bytes]:
    risk_cells: set[bytes] = set()
    
    for cell_key in cell_keys:
        level, global_id = _decode_cell_key(cell_key)
        cols = meta_level_info[level]['width']
        rows = meta_level_info[level]['height']
        
        global_u = global_id % cols
        global_v = global_id // cols
        
        # Check top edge with tCell
        t_cell = _get_cell_from_uv(level, cols, rows, global_u, global_v + 1, meta_level_info)
        if t_cell:
            if _check_risk_along_edge(risk_threshold, cell_keys, subdivide_rules, meta_level_info, level, t_cell[0], t_cell[1], ADJACENT_CHECK_NORTH):
                risk_cells.add(cell_key)
                continue
        # Check left edge with lCell
        l_cell = _get_cell_from_uv(level, cols, rows, global_u - 1, global_v, meta_level_info)
        if l_cell:
            if _check_risk_along_edge(risk_threshold, cell_keys, subdivide_rules, meta_level_info, level, l_cell[0], l_cell[1], ADJACENT_CHECK_WEST):
                risk_cells.add(cell_key)
                continue
        # Check bottom edge with bCell
        b_cell = _get_cell_from_uv(level, cols, rows, global_u, global_v - 1, meta_level_info)
        if b_cell:
            if _check_risk_along_edge(risk_threshold, cell_keys, subdivide_rules, meta_level_info, level, b_cell[0], b_cell[1], ADJACENT_CHECK_SOUTH):
                risk_cells.add(cell_key)
                continue
        # Check right edge with rCell
        r_cell = _get_cell_from_uv(level, cols, rows, global_u + 1, global_v, meta_level_info)
        if r_cell:
            if _check_risk_along_edge(risk_threshold, cell_keys, subdivide_rules, meta_level_info, level, r_cell[0], r_cell[1], ADJACENT_CHECK_EAST):
                risk_cells.add(cell_key)
                continue
    return risk_cells

def _refine_risk_cells(risk_cells: set[bytes], subdivide_rules: list[list[int]], meta_level_info: list[dict[str, int]]) -> set[bytes]:
    """
    Refine risk cells to their children cells by one level deeper
    """
    refined_cells: set[bytes] = set()
    for cell_key in risk_cells:
        level, global_id = _decode_cell_key(cell_key)
        children_global_ids = _get_children_global_ids(level, global_id, meta_level_info, subdivide_rules)
        child_level = level + 1
        for child_global_id in children_global_ids:
            child_key = _encode_cell_key(child_level, child_global_id)
            refined_cells.add(child_key)
    return refined_cells


def _find_neighbours_along_edge(
    grid_cache: GridCache,
    subdivide_rules: list[list[int]],
    meta_level_info: list[dict[str, int]],
    cell_level: int, cell_global_id: int,
    neighbour_level: int, neighbour_global_id: int,
    edge_code: EdgeCode, adjacent_check_func: Callable
):
    # Check if neighbour cell is activated (whether if this cell is a leaf node)
    if grid_cache.has_cell(neighbour_level, neighbour_global_id):
        _update_cell_neighbour(grid_cache, cell_level, cell_global_id, neighbour_level, neighbour_global_id, edge_code)
    else:
        adj_children: list[tuple[int, int]] = []
        cell_stack: list[tuple[int, int]] = [(neighbour_level, neighbour_global_id)]
        
        while cell_stack:
            _level, _global_id = cell_stack.pop()
            if _level >= len(subdivide_rules):
                continue
            
            sub_width, sub_height = subdivide_rules[_level]
            children_global_ids = _get_children_global_ids(_level, _global_id, meta_level_info, subdivide_rules)
            if children_global_ids is None:
                continue
            
            for child_local_id, child_global_id in enumerate(children_global_ids):
                is_adjacent = adjacent_check_func(child_local_id, sub_width, sub_height)
                if not is_adjacent:
                    continue
                
                child_level = _level + 1
                if grid_cache.has_cell(child_level, child_global_id):
                    adj_children.append((child_level, child_global_id))
                else:
                    cell_stack.append((child_level, child_global_id))
        
        for child_level, child_global_id in adj_children:
            _update_cell_neighbour(grid_cache, cell_level, cell_global_id, child_level, child_global_id, edge_code)
        
def _find_cell_neighbours(grid_cache: GridCache, subdivide_rules: list[list[int]], meta_level_info: list[dict[str, int]]) -> set[bytes]:
    for level, global_id in grid_cache.array:
        cols = meta_level_info[level]['width']
        rows = meta_level_info[level]['height']
        
        global_u = global_id % cols
        global_v = global_id // cols
        
        # Check top edge with tCell
        t_cell = _get_cell_from_uv(level, cols, rows, global_u, global_v + 1, meta_level_info)
        if t_cell:
            _find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, t_cell[0], t_cell[1], EdgeCode.NORTH, ADJACENT_CHECK_NORTH)
        # Check left edge with lCell
        l_cell = _get_cell_from_uv(level, cols, rows, global_u - 1, global_v, meta_level_info)
        if l_cell:
            _find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, l_cell[0], l_cell[1], EdgeCode.WEST, ADJACENT_CHECK_WEST)
        # Check bottom edge with bCell
        b_cell = _get_cell_from_uv(level, cols, rows, global_u, global_v - 1, meta_level_info)
        if b_cell:
            _find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, b_cell[0], b_cell[1], EdgeCode.SOUTH, ADJACENT_CHECK_SOUTH)
        # Check right edge with rCell
        r_cell = _get_cell_from_uv(level, cols, rows, global_u + 1, global_v, meta_level_info)
        if r_cell:
            _find_neighbours_along_edge(grid_cache, subdivide_rules, meta_level_info, level, global_id, r_cell[0], r_cell[1], EdgeCode.EAST, ADJACENT_CHECK_EAST)

def _simplify_fraction(n: int, m: int) -> list[int]:
    """Find the greatest common divisor of two numbers"""
    a, b = n, m
    while b != 0:
        a, b = b, a % b
    return [n // a, m // a]

def _get_fractional_coords(level: int, global_id: int, meta_level_info: list[dict[str, int]]) -> tuple[list[int], list[int], list[int], list[int]]:
    width = meta_level_info[level]['width']
    height = meta_level_info[level]['height']
    
    u = global_id % width
    v = global_id // width
    
    x_min_frac = _simplify_fraction(u, width)
    x_max_frac = _simplify_fraction(u + 1, width)
    y_min_frac = _simplify_fraction(v, height)
    y_max_frac = _simplify_fraction(v + 1, height)
    
    return x_min_frac, x_max_frac, y_min_frac, y_max_frac

def _get_edge_index(
    cell_key_a: int, cell_key_b: int | None, 
    direction: int, edge_range_info: list[list[int]], code_from_a: EdgeCode,
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_cell_indices: list[list[int | None]]
) -> bytes:
    if direction not in (0, 1):
        raise ValueError('Direction must be either 0 (vertical) or 1 (horizontal)')
    if not isinstance(edge_range_info, list) or len(edge_range_info) != 3:
        raise ValueError('edge_range_info must be a list of three [numerator, denominator] pairs')
    
    # Unpack the range components
    # Each is expected to be a UINT32
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

        cells = [cell_key_b, cell_key_a] if code_from_a == EdgeCode.NORTH or code_from_a == EdgeCode.WEST else [cell_key_a, cell_key_b]
        edge_adj_cell_indices.append(cells)
        return edge_index
    else:
        return edge_index_dict[edge_key]
   
def _add_edge_to_cell(
    grid_cache: GridCache, cell_key: int,
    edge_code: EdgeCode, edge_index: int
):
    grid_cache.edges[cell_key][edge_code].add(edge_index)

def _calc_horizontal_edges(
    grid_cache: GridCache,
    cell_index: int, level: int,
    neighbour_indices: list[int],
    edge_code: EdgeCode, op_edge_code: EdgeCode,
    shared_y_f: list[int],
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_cell_indices: list[list[int | None]]
):
    cell_x_min_f, cell_x_max_f, _, _ = grid_cache.fract_coords[cell_index]
    cell_x_min, cell_x_max = cell_x_min_f[0] / cell_x_min_f[1], cell_x_max_f[0] / cell_x_max_f[1]
    
    # Case when no neighbour ############################################################################
    if not neighbour_indices:
        edge_index = _get_edge_index(cell_index, None, 1, [cell_x_min_f, cell_x_max_f, shared_y_f], edge_code, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
        return
    
    # Case when neighbour has lower level ###############################################################
    if len(neighbour_indices) == 1 and grid_cache.array[neighbour_indices[0]][0] < level:
        edge_index = _get_edge_index(cell_index, neighbour_indices[0], 1, [cell_x_min_f, cell_x_max_f, shared_y_f], edge_code, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
        _add_edge_to_cell(grid_cache, neighbour_indices[0], op_edge_code, edge_index)
        return
    
    # Case when neighbours have equal or higher levels ##################################################
    processed_neighbours = []
    for neighbour_index in neighbour_indices:
        n_x_min_f, n_x_max_f, _, _ = grid_cache.fract_coords[neighbour_index]
        processed_neighbours.append({
            'index': neighbour_index,
            'x_min_f': n_x_min_f,
            'x_max_f': n_x_max_f,
            'x_min': n_x_min_f[0] / n_x_min_f[1],
            'x_max': n_x_max_f[0] / n_x_max_f[1],
        })
        
    # Sort neighbours by their x_min
    processed_neighbours.sort(key=lambda n: n['x_min'])

    # Calculate edge between grid xMin and first neighbour if existed
    if cell_x_min != processed_neighbours[0]['x_min']:
        edge_index = _get_edge_index(
            cell_index, None, 1,
            [cell_x_min_f, processed_neighbours[0]['x_min_f'], shared_y_f], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_cell_indices
        )
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
    
    # Calculate edges between neighbours
    for i in range(len(processed_neighbours) - 1):
        neighbour_from = processed_neighbours[i]
        neighbour_to = processed_neighbours[i + 1]
        
        # Calculate edge of neighbour_from
        edge_index = _get_edge_index(
            cell_index, neighbour_from['index'], 1,
            [neighbour_from['x_min_f'], neighbour_from['x_max_f'], shared_y_f], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_cell_indices
        )
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
        _add_edge_to_cell(grid_cache, neighbour_from['index'], op_edge_code, edge_index)
        
        # Calculate edge between neighbourFrom and neighbourTo if existed
        if neighbour_from['x_max'] != neighbour_to['x_min']:
            edge_index = _get_edge_index(
                cell_index, None, 1,
                [neighbour_from['x_max_f'], neighbour_to['x_min_f'], shared_y_f], edge_code,
                edge_index_cache, edge_index_dict, edge_adj_cell_indices
            )
            _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
            
    # Calculate edge of last neighbour
    neighbour_last = processed_neighbours[-1]
    edge_index = _get_edge_index(
        cell_index, neighbour_last['index'], 1,
        [neighbour_last['x_min_f'], neighbour_last['x_max_f'], shared_y_f], edge_code,
        edge_index_cache, edge_index_dict, edge_adj_cell_indices
    )
    _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
    _add_edge_to_cell(grid_cache, neighbour_last['index'], op_edge_code, edge_index)
    
    # Calculate edge between last neighbour and grid xMax if existed
    if cell_x_max != neighbour_last['x_max']:
        edge_index = _get_edge_index(
            cell_index, None, 1,
            [neighbour_last['x_max_f'], cell_x_max_f, shared_y_f], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_cell_indices
        )
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)

def _calc_vertical_edges(
    grid_cache: GridCache,
    cell_index: int, level: int,
    neighbour_indices: list[int],
    edge_code: EdgeCode, op_edge_code: EdgeCode,
    shared_x_f: list[int],
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_cell_indices: list[list[int | None]]
):
    _, _, cell_y_min_f, cell_y_max_f = grid_cache.fract_coords[cell_index]
    cell_y_min, cell_y_max = cell_y_min_f[0] / cell_y_min_f[1], cell_y_max_f[0] / cell_y_max_f[1]
    
    # Case when no neighbour ############################################################################
    if not neighbour_indices:
        edge_index = _get_edge_index(cell_index, None, 0, [cell_y_min_f, cell_y_max_f, shared_x_f], edge_code, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
        return
    
    # Case when neighbour has lower level ###############################################################
    if len(neighbour_indices) == 1 and grid_cache.array[neighbour_indices[0]][0] < level:
        edge_index = _get_edge_index(cell_index, neighbour_indices[0], 0, [cell_y_min_f, cell_y_max_f, shared_x_f], edge_code, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
        _add_edge_to_cell(grid_cache, neighbour_indices[0], op_edge_code, edge_index)
        return
    
    # Case when neighbours have equal or higher levels ##################################################
    processed_neighbours = []
    for neighbour_index in neighbour_indices:
        _, _, n_y_min_f, n_y_max_f = grid_cache.fract_coords[neighbour_index]
        processed_neighbours.append({
            'index': neighbour_index,
            'y_min_f': n_y_min_f,
            'y_max_f': n_y_max_f,
            'y_min': n_y_min_f[0] / n_y_min_f[1],
            'y_max': n_y_max_f[0] / n_y_max_f[1],
        })

    # Sort neighbours by their y_min
    processed_neighbours.sort(key=lambda n: n['y_min'])

    # Calculate edge between grid yMin and first neighbour if existed
    if cell_y_min != processed_neighbours[0]['y_min']:
        edge_index = _get_edge_index(
            cell_index, None, 0,
            [cell_y_min_f, processed_neighbours[0]['y_min_f'], shared_x_f], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_cell_indices
        )
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
    
    # Calculate edges between neighbours
    for i in range(len(processed_neighbours) - 1):
        neighbour_from = processed_neighbours[i]
        neighbour_to = processed_neighbours[i + 1]
        
        # Calculate edge of neighbour_from
        edge_index = _get_edge_index(
            cell_index, neighbour_from['index'], 0,
            [neighbour_from['y_min_f'], neighbour_from['y_max_f'], shared_x_f], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_cell_indices
        )
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
        _add_edge_to_cell(grid_cache, neighbour_from['index'], op_edge_code, edge_index)
        
        # Calculate edge between neighbourFrom and neighbourTo if existed
        if neighbour_from['y_max'] != neighbour_to['y_min']:
            edge_index = _get_edge_index(
                cell_index, None, 0,
                [neighbour_from['y_max_f'], neighbour_to['y_min_f'], shared_x_f], edge_code,
                edge_index_cache, edge_index_dict, edge_adj_cell_indices
            )
            _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
            
    # Calculate edge of last neighbour
    neighbour_last = processed_neighbours[-1]
    edge_index = _get_edge_index(
        cell_index, neighbour_last['index'], 0,
        [neighbour_last['y_min_f'], neighbour_last['y_max_f'], shared_x_f], edge_code,
        edge_index_cache, edge_index_dict, edge_adj_cell_indices
    )
    _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
    _add_edge_to_cell(grid_cache, neighbour_last['index'], op_edge_code, edge_index)

    # Calculate edge between last neighbour and grid yMax if existed
    if cell_y_max != neighbour_last['y_max']:
        edge_index = _get_edge_index(
            cell_index, None, 0,
            [neighbour_last['y_max_f'], cell_y_max_f, shared_x_f], edge_code,
            edge_index_cache, edge_index_dict, edge_adj_cell_indices
        )
        _add_edge_to_cell(grid_cache, cell_index, edge_code, edge_index)
            
def _calc_cell_edges(
    grid_cache: GridCache,
    meta_level_info: list[dict[str, int]],
    edge_index_cache: list[bytes],
    edge_index_dict: dict[int, bytes],
    edge_adj_cell_indices: list[list[int | None]]
):
    # Pre-calculate fractional coordinates for each cell
    for level, global_id in grid_cache.array:
        grid_cache.fract_coords.append(_get_fractional_coords(level, global_id, meta_level_info))

    for grid_index, (level, global_id) in enumerate(grid_cache.array):
        neighbours = grid_cache.neighbours[grid_index]
        grid_x_min_frac, grid_x_max_frac, grid_y_min_frac, grid_y_max_frac = grid_cache.fract_coords[grid_index]
        
        north_neighbours = list(neighbours[EdgeCode.NORTH])
        _calc_horizontal_edges(grid_cache, grid_index, level, north_neighbours, EdgeCode.NORTH, EdgeCode.SOUTH, grid_y_max_frac, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
        
        west_neighbours = list(neighbours[EdgeCode.WEST])
        _calc_vertical_edges(grid_cache, grid_index, level, west_neighbours, EdgeCode.WEST, EdgeCode.EAST, grid_x_min_frac, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
        
        south_neighbours = list(neighbours[EdgeCode.SOUTH])
        _calc_horizontal_edges(grid_cache, grid_index, level, south_neighbours, EdgeCode.SOUTH, EdgeCode.NORTH, grid_y_min_frac, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
        
        east_neighbours = list(neighbours[EdgeCode.EAST])
        _calc_vertical_edges(grid_cache, grid_index, level, east_neighbours, EdgeCode.EAST, EdgeCode.WEST, grid_x_max_frac, edge_index_cache, edge_index_dict, edge_adj_cell_indices)

def _get_cell_coordinates(level: int, global_id: int, bbox: list[float], meta_level_info: list[dict[str, int]], grid_info: list[list[float]]) -> tuple[float, float, float, float]:
    width = meta_level_info[level]['width']
    
    u = global_id % width
    v = global_id // width
    grid_width, grid_height = grid_info[level-1]
    
    min_xs = bbox[0] + u * grid_width
    min_ys = bbox[1] + v * grid_height
    max_xs = min_xs + grid_width
    max_ys = min_ys + grid_height
    return min_xs, min_ys, max_xs, max_ys

def _generate_cell_record(
    index: int, key: bytes, edges: list[set[int]], bbox: list[float],
    meta_level_info: list[dict[str, int]], grid_info: list[list[float]]
) -> bytearray:
    level, global_id = struct.unpack('>BQ', key)
    min_xs, min_ys, max_xs, max_ys = _get_cell_coordinates(level, global_id, bbox, meta_level_info, grid_info)

    unpacked_info = [
        index + 1,                                                              # index (1-based)
        min_xs, min_ys, max_xs, max_ys,                                         # grid coordinates
        len(edges[EdgeCode.WEST]),                                              # west edge count
        len(edges[EdgeCode.EAST]),                                              # east edge count
        len(edges[EdgeCode.SOUTH]),                                             # south edge count
        len(edges[EdgeCode.NORTH]),                                             # north edge count
        *[edge_index + 1 for edge_index in sorted(edges[EdgeCode.WEST])],       # west edge indices (1-based)
        *[edge_index + 1 for edge_index in sorted(edges[EdgeCode.EAST])],       # east edge indices (1-based)
        *[edge_index + 1 for edge_index in sorted(edges[EdgeCode.SOUTH])],      # south edge indices (1-based)
        *[edge_index + 1 for edge_index in sorted(edges[EdgeCode.NORTH])],      # north edge indices (1-based)
    ]
    
    unpacked_info_type = [
        'Q',                                    # index (uint64)
        'd', 'd', 'd', 'd',                     # cell coordinates (double)
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

def _batch_cell_records_worker(
    args: tuple[bytes, list[list[set[int]]]], bbox: list[float],
    meta_level_info: list[dict[str, int]], grid_info: list[list[float]]
) -> bytearray:
    cell_data, cell_edges, offset = args

    records = bytearray()
    cell_count = len(cell_data) // 9 # each cell has 9 bytes (level: uint8 + global_id: uint64)
    for i in range(cell_count):
        start = i * 9
        end = start + 9
        key = cell_data[start:end]
        
        # Get edges for this cell
        edges = cell_edges[i]
        
        # Generate cell record
        record =  _generate_cell_record(offset + i, key, edges, bbox, meta_level_info, grid_info)
        length_prefix = struct.pack('!I', len(record)) 
        
        records += length_prefix
        records += record

    return records

def _record_cell_topology(
    grid_cache: GridCache,
    meta_bounds: list[float],
    meta_level_info: list[dict[str, int]],
    grid_info: list[list[float]],
    grid_record_path: str
):
    batch_size = 10000
    batch_args = [
        (grid_cache.slice_cells(i, batch_size), grid_cache.slice_edges(i, batch_size), i)
        for i in range(0, len(grid_cache), batch_size)
    ]
    batch_func = partial(
        _batch_cell_records_worker,
        bbox=meta_bounds,
        meta_level_info=meta_level_info,
        grid_info=grid_info
    )
    
    num_processes = min(os.cpu_count(), len(batch_args))
    with mp.Pool(processes=num_processes) as pool:
        cell_records_list = pool.map(batch_func, batch_args)
    cell_records = bytearray()
    for cell_records_chunk in cell_records_list:
        cell_records += cell_records_chunk
    
    with open(grid_record_path, 'wb') as f:
        f.write(cell_records)

def _slice_edge_info(
    start_index: int, length: int,
    edge_index_cache: list[bytes],
    edge_adj_cell_indices: list[list[int | None]]
) -> tuple[list[bytes], list[list[int | None]]]:
    if start_index < 0 or start_index >= len(edge_index_cache):
        raise IndexError('Start index out of range')
    end_index = min(start_index + length, len(edge_index_cache))
    edge_indices = edge_index_cache[start_index:end_index]
    edge_adj_cell_indices = edge_adj_cell_indices[start_index:end_index]
    return edge_indices, edge_adj_cell_indices

def _generate_edge_record(index: int, edge_data: bytes, edge_grids: list[int | None], bbox: list[float]) -> bytearray:
    direction, min_num, min_den, max_num, max_den, shared_num, shared_den = struct.unpack('!BIIIIII', edge_data)
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    if direction == 0:      # vertical edge
        x_min = bbox[0] + (shared_num / shared_den) * (bbox[2] - bbox[0])
        x_max = x_min
        y_min = bbox[1] + (min_num / min_den) * (bbox[3] - bbox[1])
        y_max = bbox[1] + (max_num / max_den) * (bbox[3] - bbox[1])
    elif direction == 1:    # horizontal edge
        x_min = bbox[0] + (min_num / min_den) * (bbox[2] - bbox[0])
        x_max = bbox[0] + (max_num / max_den) * (bbox[2] - bbox[0])
        y_min = bbox[1] + (shared_num / shared_den) * (bbox[3] - bbox[1])
        y_max = y_min
    
    return struct.pack(
        '!QBddddQQ',
        index + 1,  # index (1-based)
        direction,
        x_min, y_min, x_max, y_max,
        edge_grids[0] + 1 if edge_grids[0] is not None else 0, # cell_index_a (1-based)
        edge_grids[1] + 1 if edge_grids[1] is not None else 0  # cell_index_b (1-based)
    )

def _batch_edge_records_worker(args: tuple[list[bytes], list[list[int | None]]], bbox: list[float]) -> bytes:
    edge_data, edge_cells, offset = args

    records = bytearray()
    edge_count = len(edge_data)
    for i in range(edge_count):
        edge = edge_data[i]
        
        record = _generate_edge_record(offset + i, edge, edge_cells[i], bbox)
        length_prefix = struct.pack('!I', len(record))
        
        records += length_prefix
        records += record

    return records

def _record_edge_topology(
    edge_index_cache: list[bytes],
    edge_adj_cell_indices: list[list[int | None]],
    meta_bounds: list[float],
    edge_record_path: str
):
    batch_size = 10000
    batch_args = [
        (*_slice_edge_info(i, batch_size, edge_index_cache, edge_adj_cell_indices), i)
        for i in range(0, len(edge_index_cache), batch_size)
    ]
    batch_func = partial(
        _batch_edge_records_worker,
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

def assembly(schema_file_path: str, patch_paths: list[str], workspace: str, risk_level_threshold: int = -1):
    # Initialization ######################################################################
    
    # Create workspace directory
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Init schema info
    schema_path = Path(schema_file_path)
    schema = json.load(open(schema_path, 'r', encoding='utf-8'))
    epsg: int = schema['epsg']
    grid_info: list[list[float]] = schema['grid_info']
    first_level_resolution: list[float] = grid_info[0]
    alignment_origin: list[float] = schema['base_point']
    
    # Init bounds from all patches
    meta_bounds = _get_bounds(patch_paths)
    
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
    
    # Find activated cells in all patches #################################################
    
    current_time = time.time()
    
    # Set activated cell key container
    # Key: uin8 level + uint64 global id
    activated_cell_keys: set[bytes] = set()
    
    # Update activated cells by each patch
    for patch_path in patch_paths:
        _update_cells_by_patch(
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
            ancestor_keys = _get_all_ancestor_keys(key, meta_level_info, subdivide_rules)
            ancestor_keys_to_remove.update(ancestor_keys)
        # Batch remove ancestor keys from activated cells in the level
        activated_cell_keys.difference_update(ancestor_keys_to_remove)
    print(f'Activated cell calculation took {time.time() - current_time:.2f} seconds')
    
    # Grading cells by risk level #########################################################
    
    # Remove low-risk cells if risk_level_threshold >= 0
    if risk_level_threshold >= 0:
        current_time = time.time()
        while True:
            risk_cells = _find_risk_cells(risk_level_threshold, activated_cell_keys, subdivide_rules, meta_level_info)
            if not risk_cells:
                break
            activated_cell_keys = _refine_risk_cells(risk_cells, meta_level_info, subdivide_rules).union(activated_cell_keys.difference(risk_cells))
        print(f'Risk cell refinement took {time.time() - current_time:.2f} seconds')
    
    # Topology construction for the grid ##################################################
    
    # Sort and concatenate activated cell keys
    keys_data = b''.join(sorted(activated_cell_keys))
    
    # Free memory
    activated_cell_keys = None
    gc.collect()
    
    # Init GridCache
    grid_cache = GridCache(keys_data)
    
    # Init edge topology containers
    edge_index_cache: list[bytes] = []
    edge_index_dict: dict[int, bytes] = {}
    edge_adj_cell_indices: list[list[int | None]] = [] # for each edge, the list of adjacent grid indices, among [grid_a, grid_b], grid_a must be the north or west grid

    # Step 1: Calculate all cell neighbours
    current_time = time.time()
    _find_cell_neighbours(grid_cache, subdivide_rules, meta_level_info)
    print(f'Cell neighbour calculation took {time.time() - current_time:.2f} seconds')

    # Step 2: Calculate all cell edges
    current_time = time.time()
    _calc_cell_edges(grid_cache, meta_level_info, edge_index_cache, edge_index_dict, edge_adj_cell_indices)
    print(f'Cell edge calculation took {time.time() - current_time:.2f} seconds')
    
    print(f'Find cells: {len(grid_cache)}')
    print(f'Find cell edges: {len(edge_index_cache)}')
    
    # Step 3: Record grid topology ########################################################
    
    # Create cell topology records
    cell_record_path = workspace / 'cell_topo.bin'
    _record_cell_topology(
        grid_cache,
        meta_bounds,
        meta_level_info,
        grid_info,
        str(cell_record_path)
    )
    
    # Create edge topology records
    edge_record_path = workspace / 'edge_topo.bin'
    _record_edge_topology(
        edge_index_cache,
        edge_adj_cell_indices,
        meta_bounds,
        str(edge_record_path)
    )
    
    # Create meta json
    meta_info = {
        'epsg': epsg,
        'bounds': meta_bounds,
        'grid_info': grid_info,
        'level_info': meta_level_info,
        'subdivide_rules': subdivide_rules,
        'alignment_origin': alignment_origin,
    }
    meta_path = workspace / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)
    