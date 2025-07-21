from pathlib import Path
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).parent.parent.parent.parent

APP_CONTEXT: dict[str, str] = {
    'current_patch': None
}

class Settings(BaseSettings):
    # Server configuration
    DEBUG: bool = True
    APP_NAME: str = 'NH Grid Server'
    TEMPLATES_DIR: str = str(ROOT_DIR / 'templates/')
    
    # Memory temp directory
    MEMORY_TEMP_DIR: str | None = None
    PRE_REMOVE_MEMORY_TEMP_DIR: bool = False
    
    # Proxy configuration
    HTTP_PROXY: str
    HTTPS_PROXY: str
    
    # Treeger meta configuration
    SCENARIO_META_PATH: str
    TREEGER_SERVER_ADDRESS: str = 'thread://gridman_bstreeger'
    
    # Grid schema related constants
    GRID_SCHEMA_DIR: str = 'resource/topo/schemas/'
    
    # Grid-related constants
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

    # Solution related constants
    SOLUTION_DIR: str = 'resource/solutions/'
    SERVER_PORT: int = 8000

    class Config:
        env_file = '.env'

settings = Settings()