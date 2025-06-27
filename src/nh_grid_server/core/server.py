import os
import subprocess
from pathlib import Path
import logging

server_process: subprocess.Popen | None = None
feature_process: subprocess.Popen | None = None

logger = logging.getLogger(__name__)

def init_working_directory():
    """Ensure the working directory structure exists for the server"""
    
    resource_path = Path(os.getcwd()) / 'resource'
    resource_path.mkdir(parents=True, exist_ok=True)
    
    schemas_path = resource_path / 'schemas'
    schemas_path.mkdir(parents=True, exist_ok=True)
    
    projects_path = resource_path / 'projects'
    projects_path.mkdir(parents=True, exist_ok=True)

def get_server_status():
    global server_process
    if server_process:
        try:
            os.kill(server_process.pid, 0)
            return 'running'
        except OSError:
            server_process = None
            return 'stopped'
    return 'not_started'