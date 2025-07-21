import os
import sys
import logging
import argparse
import c_two as cc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    from crms.grid import Grid
    
    parser = argparse.ArgumentParser(description='Grid Launcher')
    parser.add_argument('--timeout', type=int, help='Timeout for the server to start (in seconds)')
    parser.add_argument('--server_address', type=str, required=True, help='Address for the server')
    parser.add_argument('--schema_path', type=str, required=True, help='Path to the schema directory')
    parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace directory')
    args = parser.parse_args()
    
    # Rename
    server_address = args.server_address
    schema_path = args.schema_path
    workspace = args.workspace

    # Init CRM
    crm = Grid(schema_path, workspace)
    
    # Launch CRM server
    logger.info('Starting Grid CRM...')
    server = cc.rpc.Server(server_address, crm)
    server.start()
    logger.info('Grid CRM started at %s', server_address)
    try:
        if server.wait_for_termination(None if (args.timeout == -1 or args.timeout == 0) else args.timeout):
            logger.info('Timeout reached, terminating Grid CRM...')
            server.stop()
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received, terminating Grid CRM...')
        server.stop()
    finally:
        logger.info('Grid CRM terminated.')