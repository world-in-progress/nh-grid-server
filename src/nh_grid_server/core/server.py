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
    
    topo_path = resource_path / 'topo'
    topo_path.mkdir(parents=True, exist_ok=True)
    
    schemas_path = topo_path / 'schemas'
    schemas_path.mkdir(parents=True, exist_ok=True)
    
    dems_path = resource_path / 'dems'
    dems_path.mkdir(parents=True, exist_ok=True)
    
    lums_path = resource_path / 'lums'
    lums_path.mkdir(parents=True, exist_ok=True)
    
    vectors_path = resource_path / 'vectors'
    vectors_path.mkdir(parents=True, exist_ok=True)
    
    rainfalls_path = resource_path / 'rainfalls'
    rainfalls_path.mkdir(parents=True, exist_ok=True)
    
    solutions_path = resource_path / 'solutions'
    solutions_path.mkdir(parents=True, exist_ok=True)
    
    instances_path = resource_path / 'simulations'
    instances_path.mkdir(parents=True, exist_ok=True)
    
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