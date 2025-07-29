import os
import sys
import logging
import argparse
import c_two as cc

# Import Common (CRM)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crms.common import Common

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Common Launcher')
    parser.add_argument('--timeout', type=int, help='Timeout for the server to start (in seconds)')
    parser.add_argument('--server_address', type=str, required=True, help='TCP address for the server')
    parser.add_argument('--name', type=str, required=True, help='Name of the common resource')
    parser.add_argument('--type', type=str, required=True, help='Type of the common resource')
    parser.add_argument('--src_path', type=str, required=True, help='Source path of the common resource')
    args = parser.parse_args()

    server_address = args.server_address

    # Init CRM
    crm = Common(
        args.name,
        args.type,
        args.src_path
    )
    
    # Launch CRM server
    logger.info('Starting CRM server...')
    server = cc.rpc.Server(server_address, crm)
    server.start()
    logger.info('CRM server started at %s', server_address)
    try:
        if server.wait_for_termination(None if (args.timeout == -1 or args.timeout == 0) else args.timeout):
            logger.info('Common timeout reached, terminating CRM...')
            server.stop()
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received, terminating CRM...')
        server.stop()
    finally:
        logger.info('CRM terminated.')