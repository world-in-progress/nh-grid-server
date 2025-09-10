import logging
import uvicorn
from src.nh_grid_server.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    import os
    import sys
    if sys.platform.startswith('win') or sys.platform.startswith('linux'):
        venv_path = sys.prefix
        os.environ['PROJ_LIB'] = os.path.join(venv_path, 'Lib', 'site-packages', 'osgeo', 'data', 'proj')

    uvicorn.run('src.nh_grid_server.main:app', host='0.0.0.0', port=settings.SERVER_PORT, reload=True)

    # import time
    # from crms.grid import Grid, IGrid
    
    # grid = Grid()
    # grid.create_meta_overview()
    
    # current_time = time.time()
    
    # grid.merge()
    
    # elapsed_time = time.time() - current_time
    # print(f"Patch processed in {elapsed_time:.2f} seconds")