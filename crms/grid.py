import os
import json
import mmap
import sqlite3
import c_two as cc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp

from pathlib import Path
from contextlib import contextmanager

from icrms.igrid import IGrid, PatchInfo
from crms.patch import Patch, GridSchema

ATTR_INDEX_KEY = 'index_key'
WORKER_PATCH_OBJ = None
WORKER_MMAP_OBJ = None
WORKER_FILE_HANDLE = None

class Overview:
    def __init__(self, size):
        self.size = size
        self.data = bytearray((size + 7) // 8)

    def set_value(self, index, value):
        if index < 0 or index >= self.size:
            raise IndexError('Index out of bounds')
        
        byte_index = index // 8
        bit_index = index % 8
        if value:
            self.data[byte_index] |= (1 << bit_index)
        else:
            self.data[byte_index] &= ~(1 << bit_index)

    def get_value(self, index):
        if index < 0 or index >= self.size:
            raise IndexError('Index out of bounds')

        byte_index = index // 8
        bit_index = index % 8
        return (self.data[byte_index] >> bit_index) & 1
    
    @property
    def binary_sequence(self) -> str:
        binary_string_parts = []
        for i in range(self.size):
            binary_string_parts.append(str(self.get_value(i)))
        return ''.join(binary_string_parts)

@cc.iicrm
class Grid(IGrid):
    # def __init__(self, workspace: str):
    #     self.path = Path(workspace)
    #     self.db_path = self.path / 'grid.db'
    #     self.path.mkdir(parents=True, exist_ok=True)
    #     self._init_db()
    
    def __init__(self):
        schema_path = Path('resource', 'topo', 'schemas', '1', 'schema.json')
        schema = json.load(open(schema_path, 'r'))
        self.epsg: int = schema['epsg']
        self.grid_info: list[list[float]] = schema['grid_info']
        self.first_size: list[float] = self.grid_info[0]
        self.first_level_width = 0
        self.first_level_height = 0
        
        self.grid_ov_path = Path('resource', 'topo', 'schemas', '1', 'grids', 'meta_overview.bin')
        self.grid_ov_path.parent.mkdir(parents=True, exist_ok=True)
        if self.grid_ov_path.exists():
            self.grid_ov_path.unlink()

        self.ov_info: list[tuple[list[int], int]] = []
        for i in range(len(self.grid_info)):
            if i == 0:
                continue
            width = int(self.grid_info[0][0] / self.grid_info[i][0])
            height = int(self.grid_info[0][1] / self.grid_info[i][1])
            self.ov_info.append((
                [width, height],
                width * height
            ))
        
        self.ov_bit_length = 1
        for info in self.ov_info:
            _, size = info
            self.ov_bit_length += size
        self.ov_byte_length = (self.ov_bit_length + 7) // 8

        self.ov_offset = [0]
        for info in self.ov_info:
            _, size = info
            self.ov_offset.append(self.ov_offset[-1] + size)
        
        inf, neg_inf = float('inf'), float('-inf')
        self.bounds = [inf, inf, neg_inf, neg_inf]  # [min_x, min_y, max_x, max_y]

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS grid_patches (
                    node_key TEXT,
                    treeger_address TEXT,
                    PRIMARY KEY (node_key, treeger_address)
                )
            """)
            conn.commit()
    
    @contextmanager
    def _connect_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def has_patch(self, patch_info: PatchInfo) -> bool:
        with self._connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM grid_patches 
                WHERE node_key = ? AND treeger_address = ?
            """, (patch_info.node_key, patch_info.treeger_address))
            return cursor.fetchone() is not None

    def add_patch(self, patch_info: PatchInfo):
        if self.has_patch(patch_info):
            return
        
        with self._connect_db() as conn:
            conn.execute("""
                INSERT INTO grid_patches (node_key, treeger_address)
                VALUES (?, ?)
            """, (patch_info.node_key, patch_info.treeger_address))
            conn.commit()

    def remove_patch(self, patch_info: PatchInfo):
        if not self.has_patch(patch_info):
            return
        
        with self._connect_db() as conn:
            conn.execute("""
                DELETE FROM grid_patches 
                WHERE node_key = ? AND treeger_address = ?
            """, (patch_info.node_key, patch_info.treeger_address))
            conn.commit()
    
    def is_not_empty(self) -> bool:
        with self._connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM grid_patches")
            count = cursor.fetchone()[0]
            return count > 0
    
    def list_patches(self) -> list[PatchInfo]:
        with self._connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT node_key, treeger_address FROM grid_patches")
            rows = cursor.fetchall()
            return [PatchInfo(node_key=row['node_key'], treeger_address=row['treeger_address']) for row in rows]
    
    def clear_patches(self):
        with self._connect_db() as conn:
            conn.execute("DELETE FROM grid_patches")
            conn.commit()
    
    def set_patches(self, patch_infos: list[PatchInfo]):
        if self.is_not_empty():
            self.clear_patches()
            
        for patch_info in patch_infos:
            self.add_patch(patch_info)
    
    @property
    def entry_ov(self)-> Overview:
        """Provide an overview from a grid at the first level"""
        return Overview(self.ov_bit_length)
    
    def create_meta_overview(self):
        tp_schema = Path('resource', 'topo', 'schemas', '1', 'schema.json')
        tp_patch = Path('resource', 'topo', 'schemas', '1', 'patches', '2')
        tps = [Patch(tp_schema, tp_patch)]
        
        # Update bounds
        for patch in tps:
            schema = patch.get_schema()
            self.bounds[0] = min(self.bounds[0], schema.bounds[0])
            self.bounds[1] = min(self.bounds[1], schema.bounds[1])
            self.bounds[2] = max(self.bounds[2], schema.bounds[2])
            self.bounds[3] = max(self.bounds[3], schema.bounds[3])
        
        # Update first level width and height
        self.first_level_width = int((self.bounds[2] - self.bounds[0]) / self.first_size[0])
        self.first_level_height = int((self.bounds[3] - self.bounds[1]) / self.first_size[1])
        
        # Create overview
        ov_width = int((self.bounds[2] - self.bounds[0]) / self.first_size[0])
        ov_height = int((self.bounds[3] - self.bounds[1]) / self.first_size[1])
        all_ov_size = ov_width * ov_height * self.ov_byte_length
        
        # Create meta overview file
        with open(self.grid_ov_path, 'wb') as f:
            f.write(bytearray(all_ov_size))

    def process_patch(self):
        schema_path = Path('resource', 'topo', 'schemas', '1', 'schema.json')
        patch_path = Path('resource', 'topo', 'schemas', '1', 'patches', '2')
        patch = Patch(schema_path, patch_path)
        first_level_size = patch.level_info[1]['width'] * patch.level_info[1]['height']
        
        task_args = [
            (
                gid,
                self.ov_offset, 
                self.ov_bit_length,
                self.bounds,
                self.first_size,
                self.first_level_width,
                self.ov_byte_length
            ) 
            for gid in range(first_level_size)
        ]
        results = []
        num_processes = os.cpu_count()
        
        with mp.Pool(processes=num_processes, initializer=_init_worker, initargs=(patch, self.grid_ov_path)) as pool:
            pool.map(_process_chunk_worker, task_args)

        # with open(self.grid_ov_path, 'r+b') as f:
        #     with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
        #         for ov_byte_offset, ov_data in results:
        #             if ov_data is None: continue
        #             meta_ov_chunk = np.frombuffer(mm, dtype=np.uint8, count=self.ov_byte_length, offset=ov_byte_offset)
        #             patch_ov_chunk = np.frombuffer(ov_data, dtype=np.uint8, count=self.ov_byte_length)
        #             np.bitwise_xor(meta_ov_chunk, patch_ov_chunk, out=meta_ov_chunk)
        #         mm.flush()
        #         del meta_ov_chunk
        #         del patch_ov_chunk
        
        # schema = patch.get_schema()
        # offset_x = int((schema.bounds[0] - self.bounds[0]) / self.first_size[0])
        # offset_y = int((schema.bounds[1] - self.bounds[1]) / self.first_size[1])
        # ov_byte_offset = (offset_y * self.first_level_width + offset_x) * self.ov_byte_length

        # for first_level_global_id in range(first_level_size):
        #     # Make overview
        #     ov = self.entry_ov
        #     p_stack = [_encode_index(1, first_level_global_id)]
        #     while p_stack:
        #         index = p_stack.pop()
        #         status = patch.get_status(index)
        #         level, global_id = _decode_index(index)
                
        #         # Active grid: update overview
        #         if status == 0b10:
        #             offset = self.ov_offset[level - 1]
        #             if level == 1:
        #                 local_id = 0
        #             else:
        #                 local_id = patch.get_local_id(level, global_id)
        #             ov.set_value(offset + local_id, True)
                
        #         # Inactive grid (not active, not deleted): check children
        #         elif status == 0b00:
        #             # Get children
        #             child_level = level + 1
        #             children_info = patch.get_children_global_ids(level, global_id)
        #             if children_info is not None:
        #                 for child_global_id in children_info:
        #                     p_stack.append(_encode_index(child_level, child_global_id))
            
        #     # Write overview to file
        #     with open(self.grid_ov_path, 'r+b') as f:
        #         with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
        #             meta_ov_chunk = np.frombuffer(mm, dtype=np.uint8, count=self.ov_byte_length, offset=ov_byte_offset)
        #             patch_ov_chunk = np.frombuffer(ov.data, dtype=np.uint8, count=self.ov_byte_length)
        #             np.bitwise_xor(meta_ov_chunk, patch_ov_chunk, out=meta_ov_chunk)
        #             mm.flush()
        #             del meta_ov_chunk
        #             del patch_ov_chunk

        print('Process patch done.')

# Helpers ##################################################

def _encode_index(level: int, global_id: int) -> np.uint64:
    """Encode level and global_id into a single index key"""
    return np.uint64(level) << 32 | np.uint64(global_id)

def _decode_index(encoded: np.uint64) -> tuple[int, int]:
    """Decode the index key into level and global_id"""
    level = int(encoded >> 32)
    global_id = int(encoded & 0xFFFFFFFF)
    return level, global_id

def _encode_index_batch(levels: np.ndarray, global_ids: np.ndarray) -> np.ndarray:
    """Encode multiple levels and global_ids into a single index key array"""
    return (levels.astype(np.uint64) << 32) | global_ids.astype(np.uint64)

def _init_worker(patch_instance: Patch, grid_ov_path: str):
    global WORKER_PATCH_OBJ, WORKER_MMAP_OBJ, WORKER_FILE_HANDLE
    WORKER_PATCH_OBJ = patch_instance
    WORKER_FILE_HANDLE = open(grid_ov_path, 'r+b')
    WORKER_MMAP_OBJ = mmap.mmap(WORKER_FILE_HANDLE.fileno(), 0, access=mmap.ACCESS_WRITE)

def _process_chunk_worker(args):
    """
    Worker function to process a single patch overview.
    """
    global WORKER_PATCH_OBJ, WORKER_MMAP_OBJ
    patch = WORKER_PATCH_OBJ
    mm = WORKER_MMAP_OBJ
    
    # Unpack the rest of the arguments
    first_level_global_id, ov_offset, ov_bit_length, bounds, first_size, first_level_width, ov_byte_length = args
    
    schema = patch.get_schema()
    patch_offset_x = int((schema.bounds[0] - bounds[0]) / first_size[0])
    patch_offset_y = int((schema.bounds[1] - bounds[1]) / first_size[1])
    ov_byte_offset = (patch_offset_y * first_level_width + patch_offset_x) * ov_byte_length

    ov = Overview(ov_bit_length)
    p_stack = [_encode_index(1, first_level_global_id)]
    
    while p_stack:
        index = p_stack.pop()
        status = patch.get_status(index)
        level, global_id = _decode_index(index)
        
        if status == 0b10:
            offset = ov_offset[level - 1]
            local_id = 0 if level == 1 else patch.get_local_id(level, global_id)
            ov.set_value(offset + local_id, True)
        elif status == 0b00:
            children_info = patch.get_children_global_ids(level, global_id)
            if children_info is not None:
                for child_global_id in children_info:
                    p_stack.append(_encode_index(level + 1, child_global_id))
    
    meta_ov_chunk = np.frombuffer(mm, dtype=np.uint8, count=ov_byte_length, offset=ov_byte_offset)
    patch_ov_chunk = np.frombuffer(ov.data, dtype=np.uint8)
    np.bitwise_xor(meta_ov_chunk, patch_ov_chunk, out=meta_ov_chunk)
    mm.flush()
    del meta_ov_chunk
    del patch_ov_chunk

    return True