import os
from pathlib import Path
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent.parent

APP_CONTEXT: dict[str, str] = {
    'current_project': None,
    'current_subproject': None
}

class Settings(BaseSettings):
    APP_NAME: str = 'NH Grid Server'
    DEBUG: bool = True
    TCP_ADDRESS: str = 'tcp://localhost:5555'
    GRID_TEMP: str = 'False'
    GRID_FILE_NAME: str = 'grids.arrow'
    SCHEMA_FILE: str = 'resource/grid/schema.json'
    SCHEMA_DIR: str = 'resource/grid/schemas/'
    PROJECT_DIR: str = 'resource/grid/projects/'
    CRM_LAUNCHER_FILE: str = 'scripts/grid_crm_launcher.py'
    GRID_PROJECT_META_FILE_NAME: str = 'project.meta.json'
    GRID_SUBPROJECT_META_FILE_NAME: str = 'grid.meta.json'
    TEMPLATES_DIR: str = str(ROOT_DIR / 'templates/')
    MCP_SERVER_SCRIPT_PATH: str = str(ROOT_DIR / 'scripts/grid_mcp_server.py')
    
    ANTHROPIC_API_KEY: str
    DEEPSEEK_API_KEY: str
    HTTP_PROXY: str
    HTTPS_PROXY: str

    # CORS
    CORS_ORIGINS: list[str] = ['*']
    CORS_HEADERS: list[str] = ['*']
    CORS_METHODS: list[str] = ['*']
    CORS_CREDENTIALS: bool = True

    class Config:
        env_file = '.env'

settings = Settings()
