import os
import sys
import math
import json
import argparse
import c_two as cc
from pathlib import Path

# Import Grid (CRM)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crms.grid import Grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid Launcher')
    parser.add_argument('--temp', type=str, default='False', help='Use temporary memory for grid')
    parser.add_argument('--tcp_address', type=str, required=True, help='TCP address for the server')
    parser.add_argument('--schema_file_path', type=str, required=True, help='Path to the schema file')
    parser.add_argument('--grid_project_path', type=str, required=True, help='Path to the resource directory of grid project')
    parser.add_argument('--meta_file_name', type=str, required=True,  help='Name of the meta information file of the grid project')
    args = parser.parse_args()
    
    # Rename
    temp = args.temp
    ipc_address = 'ipc:///tmp/grid' # default address based on IPC, only can be used in Linux / MacOS
    tcp_address = args.tcp_address
    schema_file_path = args.schema_file_path
    grid_project_path = args.grid_project_path
    meta_file_name = args.meta_file_name
    
    # Get info from schema file
    schema = json.load(open(schema_file_path, 'r'))
    epsg: int = schema['epsg']
    grid_info: list[list[float]] = schema['grid_info']
    first_size: list[float] = grid_info[0]
    
    # Get info from project meta file
    meta_file = Path(grid_project_path, meta_file_name)
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
    
    # Init CRM
    crm = Grid(
        epsg, bounds, first_size, subdivide_rules, 
    )
    
    # Launch CRM server
    server = cc.message.Server(tcp_address, crm)
    server.start()
    print('CRM server started at', tcp_address)
    server.wait_for_termination()
