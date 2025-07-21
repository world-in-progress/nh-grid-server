import os
import math
import json
import logging
import c_two as cc
import numpy as np
import pandas as pd
import pyarrow as pa
from pathlib import Path
import pyarrow.parquet as pq
from collections import Counter
from icrms.ipatch import IPatch, GridSchema, GridAttribute, TopoSaveInfo

logger = logging.getLogger(__name__)

# Const ##############################

ATTR_DELETED = 'deleted'
ATTR_ACTIVATE = 'activate'
ATTR_INDEX_KEY = 'index_key'

GRID_SCHEMA: pa.Schema = pa.schema([
    (ATTR_DELETED, pa.bool_()),
    (ATTR_ACTIVATE, pa.bool_()), 
    (ATTR_INDEX_KEY, pa.uint64())
])

@cc.iicrm
class Patch(IPatch):
    """
    CRM
    =
    The Grid Resource.  
    Grid is a 2D grid system that can be subdivided into smaller grids by pre-declared subdivide rules.  
    """
    def __init__(self, schema_file_path: str, grid_patch_path: str):
        """Method to initialize Grid

        Args:
            schema_file_path (str): Path to the schema file
            grid_patch_path (str): Path to the resource directory of grid patch
        """
        # Get info from schema file
        schema = json.load(open(schema_file_path, 'r'))
        epsg: int = schema['epsg']
        grid_info: list[list[float]] = schema['grid_info']
        first_size: list[float] = grid_info[0]
        
        # Get info from patch meta file
        meta_file = Path(grid_patch_path, 'patch.meta.json')
        meta = json.load(open(meta_file, 'r'))
        bounds: list[float] = meta['bounds']
        
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
        
        # Initialize attributes
        self.epsg: int = epsg
        self.grid_info = grid_info
        self.bounds: list = bounds
        self.first_size: list[float] = first_size
        self.subdivide_rules: list[list[int]] = subdivide_rules
        self.grid_file_path = Path(grid_patch_path, 'patch.topo.parquet')
        self.grids = pd.DataFrame(columns=[ATTR_DELETED, ATTR_ACTIVATE, ATTR_INDEX_KEY])
        
        # Calculate level info for later use
        self.level_info: list[dict[str, int]] = [{'width': 1, 'height': 1}]
        for level, rule in enumerate(subdivide_rules[:-1]):
            prev_width, prev_height = self.level_info[level]['width'], self.level_info[level]['height']
            self.level_info.append({
                'width': prev_width * rule[0],
                'height': prev_height * rule[1]
            })

    def _load_from_file(self):
        """Load grid data from file streaming

        Args:
            batch_size (int): number of records processed per batch
        """
        
        try:
            if self.grid_file_path and os.path.exists(self.grid_file_path):
                grid_table = pq.read_table(self.grid_file_path)
                grid_df = grid_table.to_pandas()
                grid_df.set_index(ATTR_INDEX_KEY, inplace=True)
                self.grids = grid_df.sort_index()
                logger.info(f'Successfully loaded {len(self.grids)} grid records from {self.grid_file_path}')
            else:
                logger.warning(f"Grid file {self.grid_file_path} not found.")
            
        except Exception as e:
            logger.error(f'Error loading grid data from file: {str(e)}')
            raise e

    def _initialize_default(self):
        """Initialize grid data (ONLY Level 1) as pandas DataFrame"""
        level = 1
        total_width = self.level_info[level]['width']
        total_height = self.level_info[level]['height']
        num_grids = total_width * total_height
        
        levels = np.full(num_grids, level, dtype=np.uint8)
        global_ids = np.arange(num_grids, dtype=np.uint32)
        encoded_indices = _encode_index_batch(levels, global_ids)
        
        grid_data = {
            ATTR_ACTIVATE: np.full(num_grids, True),
            ATTR_DELETED: np.full(num_grids, False, dtype=np.bool_),
            ATTR_INDEX_KEY: encoded_indices
        }

        df = pd.DataFrame(grid_data)
        df.set_index([ATTR_INDEX_KEY], inplace=True)

        self.grids = df
        print(f'Successfully initialized grid data with {num_grids} grids at level 1')
   
    def _load_patch(self):
        """Lazy load patch data from file or initialize default"""
        if self.grids.empty:
            # Load from Parquet file if file exists
            if self.grid_file_path.exists():
                try:
                    # Load patch data from Parquet file
                    self._load_from_file()
                except Exception as e:
                    logger.error(f'Failed to load patch data from file: {str(e)}, the grid will be initialized using default method')
                    self._initialize_default()
            else:
                # Initialize patch data using default method
                logger.warning('Grid file does not exist, initializing default patch data...')
                self._initialize_default()
                logger.info('Successfully initialized default patch data')
            logger.info('Patch initialized successfully')

    def _save(self) -> dict[str, str | bool]:
        self._load_patch()
        
        grid_save_success = True
        grid_save_message = 'No grid data to save or no path provided.'

        # --- Save Grid Data ---
        if self.grid_file_path and not self.grids.empty:
            try:
                grid_reset = self.grids.reset_index(drop=False)
                grid_table = pa.Table.from_pandas(grid_reset, schema=GRID_SCHEMA)
                pq.write_table(grid_table, self.grid_file_path)
                grid_save_message = f'Successfully saved grid data to {self.grid_file_path}'
            except Exception as e:
                grid_save_success = False
                grid_save_message = f'Failed to save grid data: {str(e)}'
        if grid_save_success:
            return {'success': True, 'message': grid_save_message}
        else:
            return {'success': False, 'message': grid_save_message}

    def terminate(self) -> bool:
        """Save the grid data to Parquet file
        Returns:
            bool: Whether the save was successful
        """
        try:
            result = self._save()
            if not result['success']:
                raise Exception(result['message'])
            logger.info(result['message'])
            return True
        except Exception as e:
            logger.error(f'Error saving data: {str(e)}')
            return False

    def save(self) -> TopoSaveInfo:
        """
        Save the grid data to an Parquet file with optimized memory usage.
        This method writes the grid dataframe to disk using Parquet format.
        It processes the data in batches to minimize memory consumption during saving.
        Returns:
            SaveInfo: An object containing:
                - 'success': Boolean indicating success (True) or failure (False)
                - 'message': A string with details about the operation result
        Error conditions:
            - Returns failure if no file path is set
            - Returns failure if the grid dataframe is empty
            - Returns failure with exception details if any error occurs during saving
        """
        save_info_dict = self._save()
        logger.info(save_info_dict['message'])
        save_info = TopoSaveInfo(
            success=save_info_dict.get('success', False),
            message=save_info_dict.get('message', '')
        )
        return save_info
    
    def get_local_id(self, level: int, global_id: int) -> int:
        self._load_patch()
        
        if level == 0:
            return global_id
        total_width = self.level_info[level]['width']
        sub_width = self.subdivide_rules[level - 1][0]
        sub_height = self.subdivide_rules[level - 1][1]
        local_x = global_id % total_width
        local_y = global_id // total_width
        return (((local_y % sub_height) * sub_width) + (local_x % sub_width))
    
    def _get_parent_global_id(self, level: int, global_id: int) -> int:
        """Method to get parent global id
        Args:
            level (int): level of provided grids
            global_id (int): global_id of provided grids
        Returns:
            parent_global_id (int): parent global id of provided grids
        """
        total_width = self.level_info[level]['width']
        sub_width = self.subdivide_rules[level - 1][0]
        sub_height = self.subdivide_rules[level - 1][1]
        u = global_id % total_width
        v = global_id // total_width
        return (v // sub_height) * self.level_info[level - 1]['width'] + (u // sub_width)
    
    def _get_coordinates(self, level: int, global_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Method to calculate coordinates for provided grids having same level
        
        Args:
            level (int): level of provided grids
            global_ids (list[int]): global_ids of provided grids

        Returns:
            coordinates (tuple[list[float], list[float], list[float], list[float]]): coordinates of provided grids, orgnized by tuple of (min_xs, min_ys, max_xs, max_ys)
        """
        bbox = self.bounds
        width = self.level_info[level]['width']
        height = self.level_info[level]['height']
        
        golbal_xs = global_ids % width
        global_ys = global_ids // width
        min_xs = bbox[0] + (bbox[2] - bbox[0]) * golbal_xs / width
        min_ys = bbox[1] + (bbox[3] - bbox[1]) * global_ys / height
        max_xs = bbox[0] + (bbox[2] - bbox[0]) * (golbal_xs + 1) / width
        max_ys = bbox[1] + (bbox[3] - bbox[1]) * (global_ys + 1) / height
        return (min_xs, min_ys, max_xs, max_ys)

    def get_children_global_ids(self, level: int, global_id: int) -> list[int] | None:
        self._load_patch()
        
        if (level < 0) or (level >= len(self.level_info)):
            return None
        
        width = self.level_info[level]['width']
        global_u = global_id % width
        global_v = global_id // width
        sub_width = self.subdivide_rules[level][0]
        sub_height = self.subdivide_rules[level][1]
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
    
    def get_schema(self) -> GridSchema:
        """Method to get grid schema

        Returns:
            GridSchema: grid schema
        """
        return GridSchema(
            epsg=self.epsg,
            bounds=self.bounds,
            first_size=self.first_size,
            subdivide_rules=self.subdivide_rules
        )

    def get_parents(self, levels: list[int], global_ids: list[int]) -> tuple[list[int], list[int]]:
        """Method to get parent keys for provided grids having same level

        Args:
            levels (list[int]): levels of provided grids
            global_ids (list[int]): global_ids of provided grids

        Returns:
            multi_parent_info (tuple[list[int], list[int]]): parent levels and global_ids of provided grids
        """
        self._load_patch()
        
        parent_set: set[tuple[int, int]] = set()
        for level, global_id in zip(levels, global_ids):
            if level == 1:
                parent_set.add((level, global_id))
                continue
            
            parent_global_id = self._get_parent_global_id(level, global_id)
            parent_set.add((level - 1, parent_global_id))
        if not parent_set:
            return ([], [])
        
        return tuple(map(list, zip(*parent_set)))
    
    def get_status(self, index: int) -> int:
        """Method to get grid status for provided grid
        Args:
            index (int): index key of provided grid, encoded by _encode_index(level, global_id)
        Returns:
            int: grid status, 0b00 for not deleted, 0b01 for deleted, 0b10 for activated, 0b11 for invalid grid
        """
        self._load_patch()
        
        try: 
            is_deleted = self.grids.at[index, ATTR_DELETED]
            is_activate = self.grids.at[index, ATTR_ACTIVATE]
            
            return (is_deleted | (is_activate << 1))
        except KeyError:
            return 0b11 # invalid grid
        
    def subdivide_grids(self, levels: list[int], global_ids: list[int]) -> tuple[list[int], list[int]]:
        """
        Subdivide grids by turning off parent grids' activate flag and activating children's activate flags
        if the parent grid is activate and not deleted.

        Args:
            levels (list[int]): Array of levels for each grid to subdivide
            global_ids (list[int]): Array of global IDs for each grid to subdivide

        Returns:
            tuple[list[int], list[int]]: The levels and global IDs of the subdivided grids.
        """
        self._load_patch()
        
        if not levels or not global_ids:
            return [], []
        
        # Get all parents
        parent_indices = _encode_index_batch(np.array(levels, dtype=np.uint8), np.array(global_ids, dtype=np.uint32))
        existing_parents = [idx for idx in parent_indices if idx in self.grids.index]
        
        if not existing_parents:
            return [], []
        
        # Filter for valid parents (activated and not deleted)
        valid_parents = self.grids.loc[existing_parents]
        valid_parents = valid_parents[(valid_parents[ATTR_ACTIVATE]) & (~valid_parents[ATTR_DELETED])]
        if valid_parents.empty:
            return [], []

        # Collect all child grid information
        total_children_count = 0
        for encoded_idx in valid_parents.index:
            level, _ = _decode_index(encoded_idx)
            rule = self.subdivide_rules[level]
            total_children_count += rule[0] * rule[1]
        
        # Pre-allocate arrays for all child data
        all_child_levels = np.empty(total_children_count, dtype=np.uint8)
        all_child_global_ids = np.empty(total_children_count, dtype=np.uint32)
        all_child_indices = np.empty(total_children_count, dtype=np.uint64)
        all_deleted = np.full(total_children_count, False, dtype=np.bool_)
        all_activate = np.full(total_children_count, True, dtype=np.bool_)
        
        # Process each parent grid
        child_index = 0
        for encoded_idx in valid_parents.index:
            level, global_id = _decode_index(encoded_idx)
            child_global_ids = self.get_children_global_ids(level, global_id)
            if not child_global_ids:
                continue
            
            child_level = level + 1
            child_count = len(child_global_ids)
            end_index = child_index + child_count
            
            all_child_levels[child_index:end_index] = child_level
            all_child_global_ids[child_index:end_index] = child_global_ids
            child_encoded_indices = _encode_index_batch(
                np.full(child_count, child_level, dtype=np.uint8),
                np.array(child_global_ids, dtype=np.uint32)
            )
            all_child_indices[child_index:end_index] = child_encoded_indices
            
            # Update the current position
            child_index = end_index
        
        # If no children were added, return early
        if child_index == 0:
            return [], []
        
        # Trim arrays to actual size used
        if child_index < total_children_count:
            all_child_levels = all_child_levels[:child_index]
            all_child_global_ids = all_child_global_ids[:child_index]
            all_child_indices = all_child_indices[:child_index]
            all_deleted = all_deleted[:child_index]
            all_activate = all_activate[:child_index]
        
        # Create data for DataFrame construction
        child_data = {
            ATTR_DELETED: all_deleted,
            ATTR_ACTIVATE: all_activate,
            ATTR_INDEX_KEY: all_child_indices
        }
        
        # Make child DataFrame
        children = pd.DataFrame(child_data, columns=[
            ATTR_DELETED, ATTR_ACTIVATE, ATTR_INDEX_KEY
        ])
        children.set_index(ATTR_INDEX_KEY, inplace=True)

        # Update existing children and add new ones
        existing_mask = children.index.isin(self.grids.index)
        
        if existing_mask.any():
            # Update existing children attributes
            existing_indices = children.index[existing_mask]
            self.grids.loc[existing_indices, ATTR_ACTIVATE] = True
            self.grids.loc[existing_indices, ATTR_DELETED] = False
            
            # Add only new children
            new_children = children.loc[~existing_mask]
            if not new_children.empty:
                self.grids = pd.concat([self.grids, new_children])
        else:
            # All children are new
            self.grids = pd.concat([self.grids, children])

        # Deactivate parent grids
        self.grids.loc[valid_parents.index, ATTR_ACTIVATE] = False

        return all_child_levels.tolist(), all_child_global_ids.tolist()
    
    def delete_grids(self, levels: list[int], global_ids: list[int]):
        """Method to delete grids.

        Args:
            levels (list[int]): levels of grids to delete
            global_ids (list[int]): global_ids of grids to delete
        """
        self._load_patch()
        
        encoded_indices = _encode_index_batch(np.array(levels, dtype=np.uint8), np.array(global_ids, dtype=np.uint32))
        existing_grids = [idx for idx in encoded_indices if idx in self.grids.index]
        
        if len(existing_grids) == 0:
            return
        
        # Filter for valid grids
        valid_grids = self.grids.loc[existing_grids]
        valid_grids = valid_grids[valid_grids[ATTR_ACTIVATE] & (~valid_grids[ATTR_DELETED])]
        if valid_grids.empty:
            return
        
        # Update deleted status
        self.grids.loc[valid_grids.index, ATTR_DELETED] = True
        self.grids.loc[valid_grids.index, ATTR_ACTIVATE] = False
    
    def get_active_grid_infos(self) -> tuple[list[int], list[int]]:
        """Method to get all active grids' global ids and levels

        Returns:
            tuple[list[int], list[int]]: active grids' global ids and levels
        """
        self._load_patch()
        
        active_grids = self.grids[self.grids[ATTR_ACTIVATE] == True]
        levels, global_ids = _decode_index_batch(active_grids.index.values)
        return levels.tolist(), global_ids.tolist()
    
    def get_deleted_grid_infos(self) -> tuple[list[int], list[int]]:
        """Method to get all deleted grids' global ids and levels

        Returns:
            tuple[list[int], list[int]]: deleted grids' global ids and levels
        """
        self._load_patch()
        
        deleted_grids = self.grids[self.grids[ATTR_DELETED] == True]
        levels, global_ids = _decode_index_batch(deleted_grids.index.values)
        return levels.tolist(), global_ids.tolist()
    
    def get_multi_grid_bboxes(self, levels: list[int], global_ids: list[int]) -> list[float]:
        """Method to get bounding boxes of multiple grids

        Args:
            levels (list[int]): levels of the grids
            global_ids (list[int]): global ids of the grids

        Returns:
            list[float]: list of bounding boxes of the grids, formatted as [grid1_min_x, grid1_min_y, grid1_max_x, grid1_max_y, grid2_min_x, grid2_min_y, grid2_max_x, grid2_max_y, ...]
        """
        if not levels or not global_ids:
            return []
        
        self._load_patch()
        
        levels_np = np.array(levels, dtype=np.uint8)
        global_ids_np = np.array(global_ids, dtype=np.uint32)
        result_array = np.empty((len(levels), 4), dtype=np.float64)
        
        # Process according to levels
        unique_levels = np.unique(levels_np)
        for level in unique_levels:
            levels_mask = levels_np == level
            current_global_ids = global_ids_np[levels_mask]
            original_indices = np.where(levels_mask)[0]
            
            min_xs, min_ys, max_xs, max_ys = self._get_coordinates(level, current_global_ids)
            result_array[original_indices] = np.column_stack((min_xs, min_ys, max_xs, max_ys))
            
        return result_array.flatten().tolist()

    def merge_multi_grids(self, levels: list[int], global_ids: list[int]) -> tuple[list[int], list[int]]:
        """Merges multiple child grids into their respective parent grid

        This operation typically deactivates the specified child grids and
        activates their common parent grid.  
        Merging is only possible if all child grids are provided.

        Args:
            levels (list[int]): The levels of the child grids to be merged.
            global_ids (list[int]): The global IDs of the child grids to be merged.

        Returns:
            tuple[list[int], list[int]]: The levels and global IDs of the activated parent grids.
        """
        if not levels or not global_ids:
            return [], []
        
        self._load_patch()
        
        # Get all parent candidates from the provided child grids
        parent_candidates: list[tuple[int, int]] = []
        for level, global_id in zip(levels, global_ids):
            if level == 1:
                continue
            else:
                parent_level = level - 1
                parent_global_id = self._get_parent_global_id(level, global_id)
                parent_candidates.append((parent_level, parent_global_id))
        if not parent_candidates:
            return [], []
        
        # Get parents indicies if all children are provided
        parent_indices_to_activate = []
        parent_count = Counter(parent_candidates)
        activated_parents: list[tuple[int, int]] = []
        for (parent_level, parent_global_id), count in parent_count.items():
            sub_width, sub_height = self.subdivide_rules[parent_level]
            expected_children_count = sub_width * sub_height
            
            if count == expected_children_count:
                encoded_idx = _encode_index(parent_level, parent_global_id)
                if encoded_idx in self.grids.index:
                    parent_indices_to_activate.append(encoded_idx)
                    activated_parents.append((parent_level, parent_global_id))

        if not activated_parents:
            return [], []
        
        # Batch activate parent grids
        if parent_indices_to_activate:
            self.grids.loc[parent_indices_to_activate, ATTR_ACTIVATE] = True
        
        # Get all children of activated parents
        children_indices_to_deactivate = []
        for parent_level, parent_global_id in activated_parents:
            child_level_of_activated_parent = parent_level + 1
            theoretical_child_global_ids = self._get_grid_children_global_ids(parent_level, parent_global_id)
            if theoretical_child_global_ids:
                for child_global_id in theoretical_child_global_ids:
                    encoded_idx = _encode_index(child_level_of_activated_parent, child_global_id)
                    if encoded_idx in self.grids.index:
                        children_indices_to_deactivate.append(encoded_idx)
        
        # Batch deactivate child grids
        if children_indices_to_deactivate:
            unique_children_indices = list(set(children_indices_to_deactivate))
            if unique_children_indices:
                 self.grids.loc[unique_children_indices, ATTR_ACTIVATE] = False
        
        result_levels, result_global_ids = zip(*activated_parents)
        return list(result_levels), list(result_global_ids)
    
    def recover_multi_grids(self, levels: list[int], global_ids: list[int]):
        """Recovers multiple deleted grids by activating them

        Args:
            levels (list[int]): The levels of the grids to be recovered.
            global_ids (list[int]): The global IDs of the grids to be recovered.
        """
        if not levels or not global_ids:
            return
        
        self._load_patch()
        
        # Get all indices to recover
        encoded_indices = _encode_index_batch(np.array(levels, dtype=np.uint8), np.array(global_ids, dtype=np.uint32))
        existing_grids = [idx for idx in encoded_indices if idx in self.grids.index]
        
        if len(existing_grids) == 0:
            return
        
        # Activate these grids
        self.grids.loc[existing_grids, ATTR_ACTIVATE] = True
        self.grids.loc[existing_grids, ATTR_DELETED] = False

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

def _decode_index_batch(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a batch of index keys into levels and global_ids"""
    levels = (encoded >> 32).astype(np.uint8)
    global_ids = (encoded & 0xFFFFFFFF).astype(np.uint32)
    return levels, global_ids