import os
from pathlib import Path
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent.parent

APP_CONTEXT: dict[str, str] = {
    'current_project': None,
    'current_patch': None
}

class Settings(BaseSettings):
    # Server configuration
    APP_NAME: str = 'NH Grid Server'
    DEBUG: bool = True
    TEMPLATES_DIR: str = str(ROOT_DIR / 'templates/')
    
    # Proxy configuration
    HTTP_PROXY: str
    HTTPS_PROXY: str
    
    # Treeger meta configuration
    TREEGER_SERVER_ADDRESS: str = 'memory://gridman_bstreeger'
    SCENARIO_META_PATH: str = str(ROOT_DIR / 'scenario.meta.yaml')

    # Patch CRM configuration
    TCP_ADDRESS: str = 'tcp://localhost:5556'
    CRM_LAUNCHER_FILE: str = 'scripts/grid_crm_launcher.py'

    # Feature CRM configuration
    FEATURE_TCP_ADDRESS: str = 'tcp://localhost:5556'
    FEATURE_LAUNCHER_FILE: str = 'scripts/feature_crm_launcher.py'

    # Feature related constants
    FEATURE_RESOURCE_POOL_META_FILE_NAME: str = 'resource_pool.meta.json'
    
    # Grid schema related constants
    GRID_SCHEMA_DIR: str = 'resource/schemas/'
    GRID_SCHEMA_FILE: str = 'resource/schema.json'
    
    # Grid project related constants
    GRID_PROJECT_DIR: str = 'resource/projects/'
    GRID_PROJECT_META_FILE_NAME: str = 'project.meta.json'
    
    # Grid-related constants
    GRID_PATCH_TEMP: str = 'False'
    GRID_PATCH_META_FILE_NAME: str = 'patch.meta.json'
    GRID_PATCH_TOPOLOGY_FILE_NAME: str = 'patch.topo.arrow'
    
    # AI MCP configuration
    DEEPSEEK_API_KEY: str
    ANTHROPIC_API_KEY: str
    MCP_SERVER_SCRIPT_PATH: str = str(ROOT_DIR / 'scripts/grid_mcp_server.py')

    # CORS
    CORS_ORIGINS: list[str] = ['*']
    CORS_HEADERS: list[str] = ['*']
    CORS_METHODS: list[str] = ['*']
    CORS_CREDENTIALS: bool = True

    class Config:
        env_file = '.env'

settings = Settings()
