import os
import logging
import c_two as cc
import numpy as np
import pandas as pd
import pyarrow as pa
from enum import IntEnum
from typing import Callable
import pyarrow.parquet as pq
from collections import Counter
from icrms.ipatch import IPatch, GridSchema, GridAttribute, TopoSaveInfo

logger = logging.getLogger(__name__)

# Const ##############################

ATTR_MIN_X = 'min_x'
ATTR_MIN_Y = 'min_y'
ATTR_MAX_X = 'max_x'
ATTR_MAX_Y = 'max_y'
ATTR_LOCAL_ID = 'local_id'

ATTR_DELETED = 'deleted'
ATTR_ACTIVATE = 'activate'

ATTR_TYPE = 'type'
ATTR_LEVEL = 'level'
ATTR_GLOBAL_ID = 'global_id'
ATTR_ELEVATION = 'elevation'

ATTR_INDEX_KEY = 'index_key'

GRID_SCHEMA: pa.Schema = pa.schema([
    (ATTR_DELETED, pa.bool_()),
    (ATTR_ACTIVATE, pa.bool_()), 
    (ATTR_INDEX_KEY, pa.uint64())
])

BITS_DIRECTION = 1          # Direction [ 0 : v | 1 : u ]
BITS_RANGE_COMPONENT = 16   # For min_numerator, min_denominator, max_numerator, max_denominator (UINT16)

EDGE_CODE_INVALID = -1

class EdgeCode(IntEnum):
    NORTH = 0b00  # 0
    WEST  = 0b01  # 1
    SOUTH = 0b10  # 2
    EAST  = 0b11  # 3

@cc.iicrm
class Patch(IPatch):
    """
    CRM
    =
    The Grid Resource.  
    Grid is a 2D grid system that can be subdivided into smaller grids by pre-declared subdivide rules.  
    """
    def __init__(self, epsg: int, bounds: list, first_size: list[float], subdivide_rules: list[list[int]], grid_file_path: str = ''):
        """Method to initialize Grid

        Args:
            epsg (int): epsg code of the grid
            bounds (list): bounding box of the grid (organized as [min_x, min_y, max_x, max_y])
            first_size (list[float]): [width, height] of the first level grid
            subdivide_rules (list[list[int]]): list of subdivision rules per level
            grid_file_path (str, optional): path to .parquet file containing grid data. If provided, grid data will be loaded from this file
        """
        self.dirty = False
        self.epsg: int = epsg
        self.bounds: list = bounds
        self.first_size: list[float] = first_size
        self.subdivide_rules: list[list[int]] = subdivide_rules
        self.grid_file_path = grid_file_path if grid_file_path != '' else None
        if self.grid_file_path:
            base, ext = os.path.splitext(self.grid_file_path)
            if base.endswith('.topo'):
                base_without_topo = base[:-5]
                self.topology_file_path = f"{base_without_topo}.edge{ext}"
            else:
                self.topology_file_path = f"{base}.topology{ext}"
        else:
            self.topology_file_path = None
        
        # Initialize grid DataFrame
        self.grids = pd.DataFrame(columns=[
            ATTR_DELETED, ATTR_ACTIVATE, ATTR_INDEX_KEY
        ])

        # Initialize grid neighbours
        self.grid_neighbours: dict[np.uint64, list[set[np.uint64]]] = {}
        self.grid_edges: dict[np.uint64, list[set[np.uint64]]] = {}

        # Initialize cache
        self._edge_index_cache: list[int] = []
        self._edge_index_dict: dict[int, int] = {}
        self._edge_adj_grids_global_ids: list[list[int]] = []

        # Initialize dirty mark
        self.dirty_mark: bool = False
        
        # Calculate level info for later use
        self.level_info: list[dict[str, int]] = [{'width': 1, 'height': 1}]
        for level, rule in enumerate(subdivide_rules[:-1]):
            prev_width, prev_height = self.level_info[level]['width'], self.level_info[level]['height']
            self.level_info.append({
                'width': prev_width * rule[0],
                'height': prev_height * rule[1]
            })
        
        self.grid_definition = {
            'epsg': epsg,
            'bounds': bounds,
            'first_size': first_size,
            'subdivide_rules': subdivide_rules
        }
        
        # Load from Parquet file if file exists
        if grid_file_path and os.path.exists(grid_file_path):
            try:
                # Load grid data from Parquet file
                self._load_grid_from_file()
            except Exception as e:
                logger.error(f'Failed to load grid data from file: {str(e)}, the grid will be initialized using default method')
                self._initialize_default_grid()
        else:
            # Initialize grid data using default method
            logger.warning('Grid file does not exist, initializing default grid data...')
            self._initialize_default_grid()
            logger.info('Successfully initialized default grid data')
        logger.info('Grid initialized successfully')
    
    def _save(self) -> dict[str, str | bool]:
        grid_save_success = True
        grid_save_message = "No grid data to save or no path provided."
        topo_save_success = True
        topo_save_message = "No topology data to save or no path provided."

        # --- Save Grid Data ---
        if self.grid_file_path and not self.grids.empty:
            try:
                # with pa.ipc.new_file(self.grid_file_path, GRID_SCHEMA) as writer:
                #     batch_size = 100000
                #     for chunk_start in range(0, len(self.grids), batch_size):
                #         chunk_end = min(chunk_start + batch_size, len(self.grids))
                #         chunk = self.grids.iloc[chunk_start:chunk_end]
                #         chunk_reset = chunk.reset_index(drop=False)
                #         table_chunk = pa.Table.from_pandas(chunk_reset, schema=GRID_SCHEMA)
                #         writer.write_table(table_chunk)
                grid_reset = self.grids.reset_index(drop=False)
                grid_table = pa.Table.from_pandas(grid_reset, schema=GRID_SCHEMA)
                pq.write_table(grid_table, self.grid_file_path)
                grid_save_message = f'Successfully saved grid data to {self.grid_file_path}'
            except Exception as e:
                grid_save_success = False
                grid_save_message = f'Failed to save grid data: {str(e)}'
        
        # # --- Save Topology Data ---
        # if self.topology_file_path:
        #     try:
        #         topology_data = self._parse_topology_for_save()
                
        #         if not topology_data['edge_key_cache'] and not topology_data['grid_edges']:
        #             topo_save_message = "Topology data is empty, skipping save."
        #         else:
        #             # with pa.ipc.new_file(self.topology_file_path, topology_data['schema']) as writer:
        #             #     writer.write_table(topology_data['table'])
        #             pq.write_table(topology_data['table'], self.topology_file_path)
        #             topo_save_message = f"Successfully saved topology data to {self.topology_file_path}"
        #             self.test_recode()

        #     except Exception as e:
        #         topo_save_success = False
        #         topo_save_message = f'Failed to save topology data: {str(e)}'

        # if grid_save_success and topo_save_success:
        if grid_save_success:
            self.dirty_mark = False
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

    # def _load_grid_from_file(self, batch_size: int = 100000):
    def _load_grid_from_file(self):
        """Load grid data from file streaming

        Args:
            batch_size (int): number of records processed per batch
        """
        
        try:
            if self.grid_file_path and os.path.exists(self.grid_file_path):
                # all_dfs = []
                # arrow_batches_buffer = []
                # current_rows_in_buffer = 0
                
                # with pa.ipc.open_file(self.grid_file_path) as reader:
                #     logger.info(f'Loading grid data from {self.grid_file_path}, Total Arrow batches: {reader.num_record_batches}')
                #     for i in range(reader.num_record_batches):
                #         batch = reader.get_batch(i)
                #         arrow_batches_buffer.append(batch)
                #         current_rows_in_buffer += batch.num_rows
                        
                #         if current_rows_in_buffer >= batch_size or (i == reader.num_record_batches - 1 and arrow_batches_buffer):
                #             if arrow_batches_buffer:
                #                 logger.debug(f'Processing {len(arrow_batches_buffer)} Arrow batches with {current_rows_in_buffer} rows.')
                #                 partial_table = pa.Table.from_batches(arrow_batches_buffer, schema=GRID_SCHEMA)
                #                 arrow_batches_buffer = []
                #                 current_rows_in_buffer = 0
                                
                #                 partial_df = partial_table.to_pandas(use_threads=True, split_blocks=True, self_destruct=True)
                #                 partial_df.set_index(ATTR_INDEX_KEY, inplace=True)
                #                 all_dfs.append(partial_df)
                #                 logger.debug(f'Append DataFrame chunk. Number of chunks: {len(all_dfs)}')
                grid_table = pq.read_table(self.grid_file_path)
                grid_df = grid_table.to_pandas()
                grid_df.set_index(ATTR_INDEX_KEY, inplace=True)
                self.grids = grid_df.sort_index()
                                
                # if all_dfs:
                #     logger.info(f'Concatenating {len(all_dfs)} DataFrame chunks...')
                #     self.grids = pd.concat(all_dfs, copy=False)
                #     self.grids = self.grids.sort_index()
                logger.info(f'Successfully loaded {len(self.grids)} grid records from {self.grid_file_path}')
            else:
                logger.warning(f"Grid file {self.grid_file_path} not found.")
            
        except Exception as e:
            logger.error(f'Error loading grid data from file: {str(e)}')
            raise e

        # Load topology data
        # try:
        #     if self.topology_file_path and os.path.exists(self.topology_file_path):
        #         logger.info(f'Loading topology data from {self.topology_file_path}')
        #         # with pa.ipc.open_file(self.topology_file_path) as reader:
        #         #     topo_table = reader.read_all()
        #         topo_table = pq.read_table(self.topology_file_path)
                
        #         self._release()

        #         if topo_table.num_rows > 0:
        #             serialized_edge_keys = topo_table['edge_key'].to_pylist()
                    
        #             self._edge_index_cache = []
        #             for serialized_key in serialized_edge_keys:
        #                 if serialized_key is not None:
        #                     direction = int.from_bytes(serialized_key[0:1], 'big')
        #                     remaining_bits = int.from_bytes(serialized_key[1:13], 'big')
        #                     edge_key = (direction << 96) | remaining_bits
        #                     self._edge_index_cache.append(edge_key)

        #             self._edge_index_dict = {key: i for i, key in enumerate(self._edge_index_cache)}
                    
        #             self._edge_adj_grids_global_ids = [item.as_py() if item.is_valid else [] for item in topo_table['adj_grids']]

        #             self.grid_edges = {}
        #             grid_indices = topo_table['grid_idx'].to_pylist()
        #             north_edges = topo_table['north_edges'].to_pylist()
        #             south_edges = topo_table['south_edges'].to_pylist()
        #             west_edges = topo_table['west_edges'].to_pylist()
        #             east_edges = topo_table['east_edges'].to_pylist()

        #             for i, grid_idx in enumerate(grid_indices):
        #                 if grid_idx is not None:
        #                     self.grid_edges[grid_idx] = [
        #                         set(north_edges[i] if north_edges[i] else []),
        #                         set(west_edges[i] if west_edges[i] else []),
        #                         set(south_edges[i] if south_edges[i] else []),
        #                         set(east_edges[i] if east_edges[i] else [])
        #                     ]

        #         logger.info(f'Successfully loaded topology data for {len(self.grid_edges)} grids.')
        #     else:
        #          logger.warning(f'Topology file not found: {self.topology_file_path}')

        except Exception as e:
            logger.error(f'Error loading topology data from file: {str(e)}')

    def _initialize_default_grid(self):
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
        self.dirty = True
        print(f'Successfully initialized grid data with {num_grids} grids at level 1')
   
    def _get_local_ids(self, level: int, global_ids: np.ndarray) -> np.ndarray:
        """Method to calculate local_ids for provided grids having same level
        
        Args:
            level (int): level of provided grids
            global_ids (list[int]): global_ids of provided grids
        
        Returns:
            local_ids (list[int]): local_ids of provided grids
        """
        if level == 0:
            return global_ids
        total_width = self.level_info[level]['width']
        sub_width = self.subdivide_rules[level - 1][0]
        sub_height = self.subdivide_rules[level - 1][1]
        local_x = global_ids % total_width
        local_y = global_ids // total_width
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
    
    def _get_subdivide_rule(self, level: int) -> tuple[int, int]:
        subdivide_rule = self.subdivide_rules[level - 1]
        return subdivide_rule[0], subdivide_rule[1]
    
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

    def _get_grid_children_global_ids(self, level: int, global_id: int) -> list[int] | None:
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

    def get_grid_infos(self, level: int, global_ids: list[int]) -> list[GridAttribute]:
        """Method to get all attributes for provided grids having same level

        Args:
            level (int): level of provided grids
            global_ids (list[int]): global_ids of provided grids

        Returns:
            grid_infos (list[GridAttribute]): grid infos organized by GridAttribute objects with attributes: 
            level, global_id, local_id, type, elevation, deleted, activate, min_x, min_y, max_x, max_y
        """
        index_keys = [(level, global_id) for global_id in global_ids]
        filtered_grids = self.grids.loc[self.grids.index.isin(index_keys)]
        if filtered_grids.empty:
            return []
        
        global_ids_np = filtered_grids[ATTR_GLOBAL_ID].to_numpy()
        local_ids = self._get_local_ids(level, global_ids_np)
        min_xs, min_ys, max_xs, max_ys = self._get_coordinates(level, global_ids_np)
        
        filtered_grids = filtered_grids.copy()
        filtered_grids[ATTR_LOCAL_ID] = local_ids
        filtered_grids[ATTR_MIN_X] = min_xs
        filtered_grids[ATTR_MIN_Y] = min_ys
        filtered_grids[ATTR_MAX_X] = max_xs
        filtered_grids[ATTR_MAX_Y] = max_ys
        
        levels, global_ids = _decode_index_batch(filtered_grids.index.values)
        self.dirty = True
        return [
            GridAttribute(
                deleted=row[ATTR_DELETED],
                activate=row[ATTR_ACTIVATE],
                type=0,
                level=levels[i],
                global_id=global_ids[i],
                local_id=row[ATTR_LOCAL_ID],
                elevation=-9999.9,
                min_x=row[ATTR_MIN_X],
                min_y=row[ATTR_MIN_Y],
                max_x=row[ATTR_MAX_X],
                max_y=row[ATTR_MAX_Y]
            )
            for i, (_, row) in enumerate(filtered_grids.iterrows())
        ]
    
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
            child_global_ids = self._get_grid_children_global_ids(level, global_id)
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
        active_grids = self.grids[self.grids[ATTR_ACTIVATE] == True]
        levels, global_ids = _decode_index_batch(active_grids.index.values)
        return levels.tolist(), global_ids.tolist()
    
    def get_deleted_grid_infos(self) -> tuple[list[int], list[int]]:
        """Method to get all deleted grids' global ids and levels

        Returns:
            tuple[list[int], list[int]]: deleted grids' global ids and levels
        """
        deleted_grids = self.grids[self.grids[ATTR_DELETED] == True]
        levels, global_ids = _decode_index_batch(deleted_grids.index.values)
        return levels.tolist(), global_ids.tolist()
    
    def get_grid_center(self, level: int, global_id: int) -> tuple[float, float]:
        """Method to get center coordinates of a grid

        Args:
            level (int): level of the grid
            global_id (int): global id of the grid

        Returns:
            tuple[float, float]: center coordinates of the grid
        """
        min_xs, min_ys, max_xs, max_ys = self._get_coordinates(level, np.array([global_id]))
        return (min_xs[0] + max_xs[0]) / 2, (min_ys[0] + max_ys[0]) / 2
    
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

    def get_multi_grid_centers(self, levels: list[int], global_ids: list[int]) -> list[tuple[float, float]]:
        """Method to get center coordinates of multiple grids

        Args:
            levels (list[int]): levels of the grids
            global_ids (list[int]): global ids of the grids

        Returns:
            list[tuple[float, float]]: list of center coordinates of the grids
        """

        # Group global_ids and their original indices by level
        level_data_map = {}
        for i, (level, global_id) in enumerate(zip(levels, global_ids)):
            if level not in level_data_map:
                level_data_map[level] = {'ids': [], 'indices': []}
            level_data_map[level]['ids'].append(global_id)
            level_data_map[level]['indices'].append(i)

        # Pre-allocate a list to store results in the original order
        results: list[tuple[float, float]] = [None] * len(levels)

        # Process each group (level)
        for level, data in level_data_map.items():
            # Convert the list of global_ids for the current level to a NumPy array
            g_ids_np = np.array(data['ids'], dtype=np.uint32)

            # Call _get_coordinates once for all global_ids in the current level group
            min_xs, min_ys, max_xs, max_ys = self._get_coordinates(level, g_ids_np)

            # Calculate center coordinates for all grids in this batch (vectorized operation)
            center_xs = (min_xs + max_xs) / 2.0
            center_ys = (min_ys + max_ys) / 2.0

            # Distribute the calculated centers back to the results list
            # according to their original indices
            for i in range(len(data['ids'])): # Iterate through items in the current group
                original_index = data['indices'][i]
                results[original_index] = (center_xs[i], center_ys[i])

        return results

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
        
        # Get all indices to recover
        encoded_indices = _encode_index_batch(np.array(levels, dtype=np.uint8), np.array(global_ids, dtype=np.uint32))
        existing_grids = [idx for idx in encoded_indices if idx in self.grids.index]
        
        if len(existing_grids) == 0:
            return
        
        # Activate these grids
        self.grids.loc[existing_grids, ATTR_ACTIVATE] = True
        self.grids.loc[existing_grids, ATTR_DELETED] = False

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
    
    def test_recode(self):
        """
        Reads a saved topology Parquet file, decodes the information,
        and writes it to a human-readable text file for verification.
        """
        if not self.topology_file_path or not os.path.exists(self.topology_file_path):
            logger.warning(f"Topology file not found: {self.topology_file_path}. Cannot perform recode test.")
            return

        txt_output_path = os.path.splitext(self.topology_file_path)[0] + '.txt'

        try:
            # with pa.ipc.open_file(self.topology_file_path) as reader:
            #     topo_table = reader.read_all()
            topo_table = pq.read_table(self.topology_file_path)

            if topo_table.num_rows == 0:
                logger.info("Topology file is empty. Nothing to recode.")
                with open(txt_output_path, 'w', encoding='utf-8') as f:
                    f.write("Topology file is empty.")
                return

            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write("--- Decoded Topology Information ---\n\n")

                f.write("--- Edge Cache ---\n")
                serialized_edge_keys = topo_table['edge_key'].to_pylist()
                
                for i, serialized_key in enumerate(serialized_edge_keys):
                    if serialized_key is not None:
                        direction = int.from_bytes(serialized_key[0:1], 'big')
                        remaining_bits_int = int.from_bytes(serialized_key[1:13], 'big')

                        shared_den = remaining_bits_int & 0xFFFF
                        shared_num = (remaining_bits_int >> 16) & 0xFFFF
                        max_den = (remaining_bits_int >> 32) & 0xFFFF
                        max_num = (remaining_bits_int >> 48) & 0xFFFF
                        min_den = (remaining_bits_int >> 64) & 0xFFFF
                        min_num = (remaining_bits_int >> 80) & 0xFFFF
                        
                        f.write(f"Edge Index {i}:\n")
                        f.write(f"  Direction: {'horizontal' if direction == 1 else 'vertical'}\n")
                        f.write(f"  Min Fractional Coord: [{min_num}, {min_den}]\n")
                        f.write(f"  Max Fractional Coord: [{max_num}, {max_den}]\n")
                        f.write(f"  Shared Fractional Coord: [{shared_num}, {shared_den}]\n\n")

                f.write("\n--- Edge to Adjacent Grids Mapping ---\n")
                adj_grids_list = topo_table['adj_grids'].to_pylist()
                for i, serialized_key in enumerate(serialized_edge_keys):
                    if serialized_key is not None:
                        adj_grids = adj_grids_list[i]
                        f.write(f"Edge Index {i} is adjacent to Global IDs: {adj_grids}\n")
                
                f.write("\n--- Grid to Edges Mapping ---\n")
                grid_indices = topo_table['grid_idx'].to_pylist()
                north_edges = topo_table['north_edges'].to_pylist()
                south_edges = topo_table['south_edges'].to_pylist()
                west_edges = topo_table['west_edges'].to_pylist()
                east_edges = topo_table['east_edges'].to_pylist()

                for i, grid_idx in enumerate(grid_indices):
                    if grid_idx is not None:
                        level, global_id = _decode_index(np.uint64(grid_idx))
                        f.write(f"\nGrid (Level: {level}, Global ID: {global_id}):\n")
                        if north_edges[i]:
                            f.write(f"  North Edges (Indices): {sorted(north_edges[i])}\n")
                        if south_edges[i]:
                            f.write(f"  South Edges (Indices): {sorted(south_edges[i])}\n")
                        if west_edges[i]:
                            f.write(f"  West Edges (Indices): {sorted(west_edges[i])}\n")
                        if east_edges[i]:
                            f.write(f"  East Edges (Indices): {sorted(east_edges[i])}\n")
            
            logger.info(f"Successfully decoded topology and saved to {txt_output_path}")

        except Exception as e:
            logger.error(f"Failed to recode topology file: {str(e)}")

    def _get_toggle_edge_code(self, code: int) -> int:
        toggle_map = {
            EdgeCode.NORTH: EdgeCode.SOUTH,
            EdgeCode.WEST: EdgeCode.EAST,
            EdgeCode.SOUTH: EdgeCode.NORTH,
            EdgeCode.EAST: EdgeCode.WEST
        }
        
        try:
            return toggle_map[code]
        except KeyError:
            print("Invalid edge code.")
            return EDGE_CODE_INVALID

    def _update_grid_neighbour(self, grid_level: int, grid_global_id: int, neighbour_level: int, neighbour_global_id: int, edge_code: EdgeCode):
        if edge_code == EDGE_CODE_INVALID: return

        grid_idx = _encode_index(grid_level, grid_global_id)
        neighbour_idx = _encode_index(neighbour_level, neighbour_global_id)
        oppo_code = self._get_toggle_edge_code(edge_code)

        if grid_idx not in self.grid_neighbours:
            self.grid_neighbours[grid_idx] = [set() for _ in range(4)] 
        if neighbour_idx not in self.grid_neighbours:
            self.grid_neighbours[neighbour_idx] = [set() for _ in range(4)]

        self.grid_neighbours[grid_idx][edge_code].add(neighbour_idx)
        self.grid_neighbours[neighbour_idx][oppo_code].add(grid_idx)

    def _find_neighbours_along_edge(self, active_grids_idx: pd.Index, grid_level: int, grid_global_id: int, neighbour_level: int, neighbour_global_id: int, edge_code: EdgeCode, adjacent_check_func: Callable):
        # Chech if neighbour grid is activated(whether if this grid is a leaf node)
        root_neighbour_encoded_idx = _encode_index(neighbour_level, neighbour_global_id)

        if root_neighbour_encoded_idx in active_grids_idx:
            self._update_grid_neighbour(grid_level, grid_global_id, neighbour_level, neighbour_global_id, edge_code)

        else:
            adj_children_infos: list[tuple[int, int]] = []
            info_stack: list[tuple[int, int]] = [(neighbour_level, neighbour_global_id)]

            while info_stack:
                _level, _global_id = info_stack.pop()

                children_global_ids = self._get_grid_children_global_ids(_level, _global_id)

                if children_global_ids is None: 
                    continue

                if _level >= len(self.subdivide_rules): 
                    continue

                sub_width, sub_height = self.subdivide_rules[_level] 

                for child_local_id, child_global_id in enumerate(children_global_ids):
                    is_adjacent = adjacent_check_func(child_local_id, sub_width, sub_height)
                    
                    if not is_adjacent: 
                        continue

                    child_level = _level + 1
                    child_encoded_idx = _encode_index(child_level, child_global_id)

                    if child_encoded_idx in active_grids_idx:
                        adj_children_infos.append((child_level, child_global_id))
                    else:
                        info_stack.append((child_level, child_global_id))

            for child_level, child_global_id in adj_children_infos:
                self._update_grid_neighbour(grid_level, grid_global_id, child_level, child_global_id, edge_code)

    def _get_grid_info_from_uv(self, level: int, u: int, v: int) -> tuple[int, int] | None:
        """Get grid info from uv coordinates"""
        if level >= len(self.level_info) or level < 0:
            return None

        width = self.level_info[level]['width']
        height = self.level_info[level]['height']

        if u < 0 or u >= width or v < 0 or v >= height:
            return None

        global_id = v * width + u
        return (level, global_id)

    def _find_grid_neighbours(self):
        active_grids_idx = self.grids[self.grids[ATTR_ACTIVATE] == True].index

        for encoded_idx in active_grids_idx:
            current_grid_level, current_grid_global_id = _decode_index(encoded_idx)

            width = self.level_info[current_grid_level]['width']

            global_u = current_grid_global_id % width
            global_v = current_grid_global_id // width

            # Check top edge with tGrid
            t_grid_info = self._get_grid_info_from_uv(current_grid_level, global_u, global_v + 1)
            if t_grid_info:
                adjacent_check_north = lambda local_id, sub_width, sub_height: local_id < sub_width
                self._find_neighbours_along_edge(active_grids_idx, current_grid_level, current_grid_global_id, t_grid_info[0], t_grid_info[1], EdgeCode.NORTH, adjacent_check_north)
        
            # --------------------------------------------------------------------------------

            # Check left edge with tGrid
            l_grid_info = self._get_grid_info_from_uv(current_grid_level, global_u - 1, global_v)
            if l_grid_info:
                adjacent_check_west = lambda local_id, sub_width, sub_height: local_id % sub_width == sub_width - 1
                self._find_neighbours_along_edge(active_grids_idx, current_grid_level, current_grid_global_id, l_grid_info[0], l_grid_info[1], EdgeCode.WEST, adjacent_check_west)

            # --------------------------------------------------------------------------------

            # Check bottom edge with tGrid
            b_grid_info = self._get_grid_info_from_uv(current_grid_level, global_u, global_v - 1)
            if b_grid_info:
                adjacent_check_south = lambda local_id, sub_width, sub_height: local_id >= sub_width * (sub_height - 1)
                self._find_neighbours_along_edge(active_grids_idx, current_grid_level, current_grid_global_id, b_grid_info[0], b_grid_info[1], EdgeCode.SOUTH, adjacent_check_south)

            # --------------------------------------------------------------------------------
            
            # Check right edge with tGrid
            r_grid_info = self._get_grid_info_from_uv(current_grid_level, global_u + 1, global_v)
            if r_grid_info:
                adjacent_check_east = lambda local_id, sub_width, sub_height: local_id % sub_width == 0
                self._find_neighbours_along_edge(active_grids_idx, current_grid_level, current_grid_global_id, r_grid_info[0], r_grid_info[1], EdgeCode.EAST, adjacent_check_east)

    def _get_grid_neighbours(self, level: int, global_id: int) -> dict[str, list[tuple[int, int]]] | None:
        grid_idx = _encode_index(level, global_id)

        if grid_idx not in self.grid_neighbours:
            return None
        
        neighbours_list_of_sets = self.grid_neighbours[grid_idx]

        result_neighbours = {
            EdgeCode.NORTH.name: [],
            EdgeCode.SOUTH.name: [],
            EdgeCode.WEST.name: [],
            EdgeCode.EAST.name: [],
        }

        for edge_code, neighbours_set in enumerate(neighbours_list_of_sets):
            edge_name = EdgeCode(edge_code).name
            decoded_neighbours = [(_decode_index(n_idx)[0], _decode_index(n_idx)[1]) for n_idx in neighbours_set]
            result_neighbours[edge_name] = decoded_neighbours

        return result_neighbours

    def parse_topology(self) -> dict:
        # Step 0: Clear all caches
        self.grid_neighbours.clear()
        self.grid_edges.clear()
        self._edge_index_cache.clear()
        self._edge_index_dict.clear()
        self._edge_adj_grids_global_ids.clear()

        # Step 1: Calculate all grid neighbours
        self._find_grid_neighbours()

        # Step 2: Calculate all grid edges
        self.calc_grid_edges()

        # Step 3: Return all topology data
        return {
            'edge_key_cache': self._edge_index_cache,
            'edge_adj_grids_global_ids': self._edge_adj_grids_global_ids,
            'grid_edges': {str(k): [list(s) for s in v] for k, v in self.grid_edges.items()}
        }
    
    def _release(self):
        self._edge_index_cache.clear()
        self._edge_index_dict.clear()
        self.grid_edges.clear()
        self._edge_adj_grids_global_ids.clear()
    
    def _parse_topology_for_save(self) -> dict:
        """
        Parses grid topology and prepares it for serialization.
        This function calculates neighbours and edges, then formats the data
        into a pyarrow Table and its schema, with custom edge_key serialization.
        """
        # Step 1: Ensure topology is up-to-date
        self.grid_neighbours.clear()
        self.grid_edges.clear()
        self._edge_index_cache.clear()
        self._edge_index_dict.clear()
        self._edge_adj_grids_global_ids.clear()

        self._find_grid_neighbours()
        self.calc_grid_edges()

        # Step 2: Prepare data for Arrow Table
        edge_keys = self._edge_index_cache
        adj_grids = self._edge_adj_grids_global_ids
        
        grid_edges_items = list(self.grid_edges.items())
        grid_indices = [item[0] for item in grid_edges_items]
        edge_sets = [item[1] for item in grid_edges_items]

        # Serialize edge_keys to 13-byte format
        serialized_edge_keys = []
        for key in edge_keys:
            if key is not None:
                direction = (key >> 96) & 1
                remaining_bits = key & ((1 << 96) - 1)
                serialized_key = direction.to_bytes(1, 'big') + remaining_bits.to_bytes(12, 'big')
                serialized_edge_keys.append(serialized_key)
            else:
                serialized_edge_keys.append(None)

        max_len = max(len(serialized_edge_keys), len(grid_indices))
        
        padded_edge_keys = serialized_edge_keys + [None] * (max_len - len(serialized_edge_keys))
        padded_adj_grids = adj_grids + [None] * (max_len - len(adj_grids))
        padded_grid_indices = grid_indices + [None] * (max_len - len(grid_indices))

        def get_edge_list(edge_code_val):
            lists = []
            for sets in edge_sets:
                lists.append(list(sets[edge_code_val]))
            return lists + [None] * (max_len - len(edge_sets))

        north_edges = get_edge_list(EdgeCode.NORTH.value)
        south_edges = get_edge_list(EdgeCode.SOUTH.value)
        west_edges = get_edge_list(EdgeCode.WEST.value)
        east_edges = get_edge_list(EdgeCode.EAST.value)

        # Step 3: Create Arrow Table and Schema
        topo_schema = pa.schema([
            pa.field('edge_key', pa.binary(13)),
            pa.field('adj_grids', pa.list_(pa.int32())),
            pa.field('grid_idx', pa.uint64()),
            pa.field('north_edges', pa.list_(pa.int64())),
            pa.field('south_edges', pa.list_(pa.int64())),
            pa.field('west_edges', pa.list_(pa.int64())),
            pa.field('east_edges', pa.list_(pa.int64())),
        ])

        topo_table = pa.Table.from_pydict({
            'edge_key': padded_edge_keys,
            'adj_grids': padded_adj_grids,
            'grid_idx': padded_grid_indices,
            'north_edges': north_edges,
            'south_edges': south_edges,
            'west_edges': west_edges,
            'east_edges': east_edges,
        }, schema=topo_schema)
        
        return {
            "table": topo_table,
            "schema": topo_schema,
            "edge_key_cache": self._edge_index_cache,
            "grid_edges": self.grid_edges
        }

    def _get_edge_key_by_info(self, level_a:int, global_id_a: int | None, level_b: int, global_id_b: int | None, direction: 0 | 1, edge_range_info: list[list[int]]) -> int:
        """
        Generates a unique index for an edge based on grid information and edge range.
        The edge is identified by a 97-bit integer key, which is mapped to a unique index.
        This ensures the same edge shared by different grids has the same index.
    
        Args:
            grid_a (GridNode | None): The first GridNode defining the edge.
            grid_b (GridNode | None): The second GridNode defining the edge (optional, for shared edges).
            direction (int): 0 for vertical edge (constant X), 1 for horizontal edge (constant Y).
            edge_range_info (list[list[int]]): Defines the edge's fractional coordinates:
                                               [[min_num, min_den], [max_num, max_den], [shared_num, shared_den]]
                                               Each number (num, den) is expected to be a UINT16.
    
        Returns:
            int: A unique index for the edge.
    
        Raises:
            ValueError: If both grid_a and grid_b are None, or if inputs are invalid.
        """
        
        if global_id_a is None and global_id_b is None:
            raise ValueError("Both grid_a and grid_b cannot be None.")
        if direction not in (0, 1):
            raise ValueError("Invalid direction. Must be 0 (vertical) or 1 (horizontal).")
        if not isinstance(edge_range_info, list) or len(edge_range_info) != 3:
            raise ValueError('edge_range_info must be a list of three [numerator, denominator] pairs')
    
        # Unpack the range components. Each is expected to be a UINT16.
        min_num, min_den = edge_range_info[0]
        max_num, max_den = edge_range_info[1]
        shared_num, shared_den = edge_range_info[2]
    
        # Ensure canonical ordering for the varying range (min <= max)
        if min_num > max_num or (min_num == max_num and min_den > max_den):
            min_num, max_num = max_num, min_num
            min_den, max_den = max_den, min_den
    
        # Construct the 97-bit bigint key
        # Bit allocation:
        # direction: 1 bit (highest)
        # min_num: 16 bits
        # min_den: 16 bits
        # max_num: 16 bits
        # max_den: 16 bits
        # shared_num: 16 bits
        # shared_den: 16 bits (lowest)
        # Total bits = 1 + 6 * 16 = 97 bits
    
        edge_key = (
            (direction << (BITS_RANGE_COMPONENT * 6)) |  # direction (1 bit) at bit 96
            (min_num << (BITS_RANGE_COMPONENT * 5)) |    # min_num (16 bits) at bit 80
            (min_den << (BITS_RANGE_COMPONENT * 4)) |    # min_den (16 bits) at bit 64
            (max_num << (BITS_RANGE_COMPONENT * 3)) |    # max_num (16 bits) at bit 48
            (max_den << (BITS_RANGE_COMPONENT * 2)) |    # max_den (16 bits) at bit 32
            (shared_num << BITS_RANGE_COMPONENT) |       # shared_num (16 bits) at bit 16
        shared_den                                       # shared_den (16 bits) at bit 0
        )
    
        # Try get edge_index
        if edge_key not in self._edge_index_dict:
            edge_index = len(self._edge_index_cache)
            self._edge_index_dict[edge_key] = edge_index
            self._edge_index_cache.append(edge_key)
    
            grids_global_ids = []
            if global_id_a is not None:
                grids_global_ids.append(global_id_a)
            if global_id_b is not None:
                grids_global_ids.append(global_id_b)
            
            self._edge_adj_grids_global_ids.append(grids_global_ids)
        else:
            edge_index = self._edge_index_dict[edge_key]

            existing_grids_list = self._edge_adj_grids_global_ids[edge_index]
            if global_id_a is not None and global_id_a not in existing_grids_list:
                existing_grids_list.append(global_id_a)
            if global_id_b is not None and global_id_b not in existing_grids_list:
                existing_grids_list.append(global_id_b)
            existing_grids_list.sort() 
    
        return edge_index
    
    def _get_fractional_coords(self, level: int, global_id: int) -> tuple[list[int], list[int], list[int], list[int]]:
        """
        Calculates the fractional coordinates [numerator, denominator] for a grid's min/max x/y.
        """
        width = self.level_info[level]['width']
        height = self.level_info[level]['height']

        global_u = global_id % width
        global_v = global_id // width

        x_min_frac = _simplifyFraction(global_u, width)
        x_max_frac = _simplifyFraction(global_u + 1, width)
        y_min_frac = _simplifyFraction(global_v, height)
        y_max_frac = _simplifyFraction(global_v + 1, height)

        return (x_min_frac, x_max_frac, y_min_frac, y_max_frac)
    
    def calc_grid_edges(self):
        active_grids_idx = self.grids[self.grids[ATTR_ACTIVATE] == True].index
        if not active_grids_idx.empty:
            for encoded_idx in active_grids_idx:
                if encoded_idx not in self.grid_neighbours:
                    continue
                neighbours_by_edge = self.grid_neighbours[encoded_idx]
                level, global_id = _decode_index(encoded_idx)
                grid_x_min_frac, grid_x_max_frac, grid_y_min_frac, grid_y_max_frac = self._get_fractional_coords(level, global_id)

                north_neighbours = [(_decode_index(n_idx)) for n_idx in neighbours_by_edge[EdgeCode.NORTH]]
                self._calc_horizontal_edges(level, global_id, north_neighbours, EdgeCode.NORTH, EdgeCode.SOUTH, grid_y_max_frac)

                south_neighbours = [(_decode_index(n_idx)) for n_idx in neighbours_by_edge[EdgeCode.SOUTH]]
                self._calc_horizontal_edges(level, global_id, south_neighbours, EdgeCode.SOUTH, EdgeCode.NORTH, grid_y_min_frac)

                west_neighbours = [(_decode_index(n_idx)) for n_idx in neighbours_by_edge[EdgeCode.WEST]]
                self._calc_vertical_edges(level, global_id, west_neighbours, EdgeCode.WEST, EdgeCode.EAST, grid_x_min_frac)

                east_neighbours = [(_decode_index(n_idx)) for n_idx in neighbours_by_edge[EdgeCode.EAST]]
                self._calc_vertical_edges(level, global_id, east_neighbours, EdgeCode.EAST, EdgeCode.WEST, grid_x_max_frac)

    def _add_grid_edge(self, level: int, global_id: int, edge_code: EdgeCode, edge_index: int):
        grid_idx = _encode_index(level, global_id)
        if grid_idx not in self.grid_edges:
            self.grid_edges[grid_idx] = [set() for _ in range(4)]
        self.grid_edges[grid_idx][edge_code].add(edge_index)
        logger.debug(f"Adding edge {edge_index} to Grid (Level: {level}, Global ID: {global_id}), Edge: {edge_code.name}")

    def _calc_horizontal_edges(self, level: int, global_id: int, neighbours: list[tuple[int, int]], edge_code: EdgeCode, op_edge_code: EdgeCode, shared_y_frac: list[int]):
        grid_x_min_frac, grid_x_max_frac, _, _ = self._get_fractional_coords(level, global_id)

        processed_neighbours = []
        for n_level, n_global_id in neighbours:
            n_encoded_idx = _encode_index(n_level, n_global_id)
            if n_encoded_idx not in self.grids.index or not self.grids.loc[n_encoded_idx, ATTR_ACTIVATE]:
                continue
            n_x_min_frac, n_x_max_frac, _, _ = self._get_fractional_coords(n_level, n_global_id)
            n_min_xs, _, _, _ = self._get_coordinates(n_level, np.array([n_global_id]))
            processed_neighbours.append({
                'level': n_level,
                'global_id': n_global_id,
                'x_min_frac': n_x_min_frac,
                'x_max_frac': n_x_max_frac,
                'x_min_abs': n_min_xs[0]
            })

        # Case when no neighbour
        if not processed_neighbours:
            edge_index = self._get_edge_key_by_info(level, global_id, None, None, 1, [grid_x_min_frac, grid_x_max_frac, shared_y_frac])
            self._add_grid_edge(level, global_id, edge_code, edge_index)
            return

        processed_neighbours.sort(key=lambda n: n['x_min_abs'])

        # Check if a single lower-level neighbour fully covers the grid's edge
        if len(processed_neighbours) == 1 and processed_neighbours[0]['level'] < level:
            n = processed_neighbours[0]

            is_contained = (
                (n['x_min_frac'][0] * grid_x_min_frac[1] <= grid_x_min_frac[0] * n['x_min_frac'][1]) and
                (n['x_max_frac'][0] * grid_x_max_frac[1] >= grid_x_max_frac[0] * n['x_max_frac'][1])
            )
            if is_contained:
                edge_index = self._get_edge_key_by_info(level, global_id, n['level'], n['global_id'], 1, [grid_x_min_frac, grid_x_max_frac, shared_y_frac])
                self._add_grid_edge(level, global_id, edge_code, edge_index)
                self._add_grid_edge(n['level'], n['global_id'], op_edge_code, edge_index)
                return

        # Process neighbours with equal or higher levels
        last_x_max_frac = grid_x_min_frac
        for i, neighbour in enumerate(processed_neighbours):
            if not (last_x_max_frac[0] * neighbour['x_min_frac'][1] == neighbour['x_min_frac'][0] * last_x_max_frac[1]):
                continue

            edge_index = self._get_edge_key_by_info(
                level, global_id, neighbour['level'], neighbour['global_id'], 1,
                [neighbour['x_min_frac'], neighbour['x_max_frac'], shared_y_frac]
            )
            self._add_grid_edge(level, global_id, edge_code, edge_index)
            self._add_grid_edge(neighbour['level'], neighbour['global_id'], op_edge_code, edge_index)

            last_x_max_frac = neighbour['x_max_frac']

        # Check if the last neighbour's x_max_frac aligns with grid_x_max_frac
        if not (last_x_max_frac[0] * grid_x_max_frac[1] == grid_x_max_frac[0] * last_x_max_frac[1]):
            edge_index = self._get_edge_key_by_info(
                level, global_id, None, None, 1, [last_x_max_frac, grid_x_max_frac, shared_y_frac]
            )
            self._add_grid_edge(level, global_id, edge_code, edge_index)

    def _calc_vertical_edges(self, level: int, global_id: int, neighbours: list[tuple[int, int]], edge_code: EdgeCode, op_edge_code: EdgeCode, shared_x_frac: list[int]):
        _, _, grid_y_min_frac, grid_y_max_frac = self._get_fractional_coords(level, global_id)

        processed_neighbours = []
        for n_level, n_global_id in neighbours:
            n_encoded_idx = _encode_index(n_level, n_global_id)
            if n_encoded_idx not in self.grids.index or not self.grids.loc[n_encoded_idx, ATTR_ACTIVATE]:
                continue
            _, _, n_y_min_frac, n_y_max_frac = self._get_fractional_coords(n_level, n_global_id)
            _, n_min_ys, _, _ = self._get_coordinates(n_level, np.array([n_global_id]))
            processed_neighbours.append({
                'level': n_level,
                'global_id': n_global_id,
                'y_min_frac': n_y_min_frac,
                'y_max_frac': n_y_max_frac,
                'y_min_abs': n_min_ys[0]
            })

        # Case when no neighbour
        if not processed_neighbours:
            edge_index = self._get_edge_key_by_info(level, global_id, None, None, 0, [grid_y_min_frac, grid_y_max_frac, shared_x_frac])
            self._add_grid_edge(level, global_id, edge_code, edge_index)
            return

        processed_neighbours.sort(key=lambda n: n['y_min_abs'])

        # Check if a single lower-level neighbour fully covers the grid's edge
        if len(processed_neighbours) == 1 and processed_neighbours[0]['level'] < level:
            n = processed_neighbours[0]
            is_contained = (
                (n['y_min_frac'][0] * grid_y_min_frac[1] <= grid_y_min_frac[0] * n['y_min_frac'][1]) and
                (n['y_max_frac'][0] * grid_y_max_frac[1] >= grid_y_max_frac[0] * n['y_max_frac'][1])
            )
            if is_contained:
                edge_index = self._get_edge_key_by_info(level, global_id, n['level'], n['global_id'], 0, [grid_y_min_frac, grid_y_max_frac, shared_x_frac])
                self._add_grid_edge(level, global_id, edge_code, edge_index)
                self._add_grid_edge(n['level'], n['global_id'], op_edge_code, edge_index)
                return

        # Process neighbours with equal or higher levels
        last_y_max_frac = grid_y_min_frac
        for i, neighbour in enumerate(processed_neighbours):
            if not (last_y_max_frac[0] * neighbour['y_min_frac'][1] == neighbour['y_min_frac'][0] * last_y_max_frac[1]):
                continue

            edge_index = self._get_edge_key_by_info(
                level, global_id, neighbour['level'], neighbour['global_id'], 0,
                [neighbour['y_min_frac'], neighbour['y_max_frac'], shared_x_frac]
            )
            self._add_grid_edge(level, global_id, edge_code, edge_index)
            self._add_grid_edge(neighbour['level'], neighbour['global_id'], op_edge_code, edge_index)

            last_y_max_frac = neighbour['y_max_frac']

        # Check if the last neighbour's y_max_frac aligns with grid_y_max_frac
        if not (last_y_max_frac[0] * grid_y_max_frac[1] == grid_y_max_frac[0] * last_y_max_frac[1]):
            edge_index = self._get_edge_key_by_info(
                level, global_id, None, None, 0, [last_y_max_frac, grid_y_max_frac, shared_x_frac]
            )
            self._add_grid_edge(level, global_id, edge_code, edge_index)

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

def _simplifyFraction(n: int, m: int) -> list[int]:
    """Find the greatest common divisor of two numbers"""
    a, b = n, m
    while b != 0:
        a, b = b, a % b
    return [n // a, m // a]
