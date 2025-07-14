import os
import sys
import math
import json
import logging
import argparse
import c_two as cc
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    from crms.patch import Patch
    
    parser = argparse.ArgumentParser(description='Grid Launcher')
    parser.add_argument('--timeout', type=int, help='Timeout for the server to start (in seconds)')
    parser.add_argument('--server_address', type=str, required=True, help='Address for the server')
    parser.add_argument('--schema_file_path', type=str, required=True, help='Path to the schema file')
    parser.add_argument('--grid_patch_path', type=str, required=True, help='Path to the resource directory of grid patch')
    args = parser.parse_args()
    
    # Rename
    ipc_address = 'ipc:///tmp/grid' # default address based on IPC, only can be used in Linux / MacOS
    server_address = args.server_address
    schema_file_path = args.schema_file_path
    grid_patch_path = args.grid_patch_path
    
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
    
    # Get grid file path
    grid_file_path = Path(grid_patch_path, 'patch.topo.arrow')
    
    # Init CRM
    crm = Patch(
        epsg, bounds, first_size, subdivide_rules, grid_file_path
    )
    
    # Launch CRM server
    logger.info('Starting Grid Patch CRM...')
    server = cc.rpc.Server(server_address, crm)
    server.start()
    logger.info('Grid Patch CRM started at %s', server_address)
    try:
        if server.wait_for_termination(args.timeout):
            logger.info('Timeout reached, terminating Grid Patch CRM...')
            server.stop()
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received, terminating Grid Patch CRM...')
        server.stop()
    finally:
        logger.info('Grid Patch CRM terminated.')