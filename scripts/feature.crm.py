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
    parser.add_argument('--tcp_address', type=str, required=True, help='TCP address for the server')
    parser.add_argument('--feature_path', type=str, required=True, help='Path to the feature file')
    args = parser.parse_args()
    
    tcp_address = args.tcp_address
    
    # Init CRM
    crm = Feature(
        args.feature_path
    )
    
    # Launch CRM server
    logger.info('Starting CRM server...')
    server = cc.message.Server(tcp_address, crm)
    server.start()
    logger.info('CRM server started at %s', tcp_address)
    server.wait_for_termination()