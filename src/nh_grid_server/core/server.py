import os
import sys
import signal
import subprocess
from pathlib import Path
from .config import settings, APP_CONTEXT
from ..schemas.project import ProjectMeta
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

def set_current_project(project_meta: ProjectMeta, patch_name: str) -> bool:
    
    # Check if current project is the same as the new one
    # If not, shut down current crm server
    if APP_CONTEXT['current_project'] == project_meta.name and APP_CONTEXT['current_patch'] == patch_name:
        return
    else:
        APP_CONTEXT['current_project'] = project_meta.name
        APP_CONTEXT['current_patch'] = patch_name
        close_current_project()

    # Check if schema is valid
    schema_file_path = Path(settings.GRID_SCHEMA_DIR) / f'{project_meta.schema_name}.json'
    if not schema_file_path.exists():
        raise FileNotFoundError(f'Schema file {schema_file_path} does not exist')
    
    # Start crm server process for this project
    global server_process
    
    # Platform-specific subprocess arguments
    kwargs = {}
    if sys.platform != 'win32':
        # Unix-specific: create new process group
        kwargs['preexec_fn'] = os.setsid
    else:
        # Windows-specific: don't open a new console window
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    
    project_path = Path(settings.GRID_PROJECT_DIR, project_meta.name)
    server_process = subprocess.Popen(
        [
            sys.executable, settings.CRM_LAUNCHER_FILE,
            '--temp', settings.GRID_PATCH_TEMP,
            '--tcp_address', settings.TCP_ADDRESS,
            '--schema_file_path', schema_file_path,
            '--grid_project_path', str(project_path / patch_name),
            '--meta_file_name', settings.GRID_PATCH_META_FILE_NAME,
        ],
        **kwargs
    )
    return server_process is not None

def close_current_project():
    """Shutdown CRM server subprocess"""
    
    global server_process
    if server_process:
        if sys.platform != 'win32':
            # Unix-specific: terminate the process group
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGINT)
            except (AttributeError, ProcessLookupError):
                server_process.terminate()
        else:
            # Windows-specific: send Ctrl+C signal and then terminate
            try:
                server_process.send_signal(signal.CTRL_C_EVENT)
            except (AttributeError, ProcessLookupError):
                server_process.terminate()
        
        try:
            server_process.wait(timeout=60)
            
        except subprocess.TimeoutExpired:
            if sys.platform != 'win32':
                try:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                except (AttributeError, ProcessLookupError):
                    server_process.kill()
            else:
                server_process.kill()
        
        server_process = None
        return True
    return False

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

def set_current_feature(project_meta: ProjectMeta, patch_name: str):

    # Check if current project is the same as the new one
    # If not, shut down current crm server
    if APP_CONTEXT['current_project'] == project_meta.name and APP_CONTEXT['current_patch'] == patch_name:
        return
    else:
        APP_CONTEXT['current_project'] = project_meta.name
        APP_CONTEXT['current_patch'] = patch_name
        close_current_project()
    
    # Start crm server process for this project
    global feature_process
    
    # Platform-specific subprocess arguments
    kwargs = {}
    if sys.platform != 'win32':
        # Unix-specific: create new process group
        kwargs['preexec_fn'] = os.setsid
    else:
        # Windows-specific: don't open a new console window
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    
    project_path = Path(settings.GRID_PROJECT_DIR, project_meta.name)

    logger.info("starting feature crm")

    feature_process = subprocess.Popen(
        [
            sys.executable, settings.FEATURE_LAUNCHER_FILE,
            '--tcp_address', settings.FEATURE_TCP_ADDRESS,
            '--feature_path', str(project_path / patch_name / "feature"),
        ],
        **kwargs
    )
    return feature_process is not None
