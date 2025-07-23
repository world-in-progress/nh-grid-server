import os
import sys
import logging
import argparse
import c_two as cc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crms.raster import Raster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solution Launcher')
    parser.add_argument('--timeout', type=int, help='Timeout for the server to start (in seconds)')
    parser.add_argument('--server_address', type=str, required=True, help='TCP address for the server')
    parser.add_argument('--name', type=str, required=True, help='Name of the dem')
    parser.add_argument('--original_tif_path', type=str, required=True, help='Path to the original TIF file')

    args = parser.parse_args()
    
    server_address = args.server_address

    crm = Raster(args.name, args.original_tif_path)
    server = cc.rpc.Server(server_address, crm)
    server.start()
    logger.info(f'Starting CRM server at {server_address}')
    try:
        if server.wait_for_termination(None if (args.timeout == -1 or args.timeout == 0) else 10000):
            logger.info('Timeout reached, terminating CRM...')
            server.stop()
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received, terminating CRM...')
        server.stop()
    finally:
        logger.info('CRM terminated.')
