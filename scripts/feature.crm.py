import os
import sys
import logging
import argparse
import c_two as cc

# Import Feature (CRM)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crms.feature import Feature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Launcher')
    parser.add_argument('--timeout', type=int, help='Timeout for the server to start (in seconds)')
    parser.add_argument('--server_address', type=str, required=True, help='TCP address for the server')
    parser.add_argument('--name', type=str, required=True, help='Name of the feature')
    parser.add_argument('--type', type=str, required=True, help='Type of the feature')
    parser.add_argument('--color', type=str, required=True, help='Color of the feature')
    parser.add_argument('--epsg', type=str, required=True, help='EPSG code of the feature')
    args = parser.parse_args()

    server_address = args.server_address

    # Init CRM
    crm = Feature(
        args.name,
        args.type,
        args.color,
        args.epsg
    )
    
    # Launch CRM server
    logger.info('Starting CRM server...')
    server = cc.rpc.Server(server_address, crm)
    server.start()
    logger.info('CRM server started at %s', server_address)
    try:
        if server.wait_for_termination(None if (args.timeout == -1 or args.timeout == 0) else args.timeout):
            logger.info('Timeout reached, terminating CRM...')
            server.stop()
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received, terminating CRM...')
        server.stop()
    finally:
        logger.info('CRM terminated.')