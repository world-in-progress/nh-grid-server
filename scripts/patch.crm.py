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
    
    
    # Init CRM
    crm = Patch(schema_file_path, grid_patch_path)
    
    # Launch CRM server
    logger.info('Starting Grid Patch CRM...')
    server = cc.rpc.Server(server_address, crm)
    server.start()
    logger.info('Grid Patch CRM started at %s', server_address)
    try:
        if server.wait_for_termination(None if (args.timeout == -1 or args.timeout == 0) else args.timeout):
            logger.info('Timeout reached, terminating Grid Patch CRM...')
            server.stop()
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received, terminating Grid Patch CRM...')
        server.stop()
    finally:
        logger.info('Grid Patch CRM terminated.')