import os
import c_two as cc
from pathlib import Path
from icrms.isolution import ISolution
import logging
from src.nh_grid_server.core.config import settings
logger = logging.getLogger(__name__)

@cc.iicrm
class Solution(ISolution):
    def __init__(self, name: str, env: dict):
        self.name = name
        self.path = Path(f'{settings.SOLUTION_DIR}{self.name}')
        self.env = env

        # Create solution directory
        self.path.mkdir(parents=True, exist_ok=True)
        # # Create ref json file
        # ref_path = self.path / 'ref.json'
        # with open(ref_path, 'w', encoding='utf-8') as f:
        #     json.dump(body.model_dump(), f, ensure_ascii=False, indent=4)

    def clone_env(self) -> dict:
        env_data = {}
        for key, value in self.env.items():
            if isinstance(value, str) and os.path.isfile(value):
                try:
                    with open(value, 'r', encoding='utf-8') as f:
                        content = f.readlines()
                except UnicodeDecodeError:
                    with open(value, 'rb') as f:
                        content = f.read()
                env_data[key] = {
                    'file_name': os.path.basename(value),
                    'content': content
                }
            else:
                env_data[key] = value
        return env_data
    
    def get_env(self) -> dict:
        return self.env

    def terminate(self) -> None:
        # Do something need to be saved
        pass