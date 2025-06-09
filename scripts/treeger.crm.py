import os
import sys
import math
import json
import logging
import argparse
import c_two as cc
from pathlib import Path

# Import CRM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crms.treeger import Treeger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treeger CRM Launcher')
    parser.add_argument('--tcp_address', type=str, required=True, help='TCP address for treeger crm server')
    parser.add_argument('--meta_path', type=str, required=True, help='Treeger meta file path')
    args = parser.parse_args()
    
    # Init CRM
    crm = Treeger(meta_path=args.meta_path)
    
    # Launch CRM server
    logger.info('Starting Treeger CRM ...')
    server = cc.message.Server(args.tcp_address, crm)
    server.start()
    logger.info('Treeger CRM started at %s', args.tcp_address)
    server.wait_for_termination()
