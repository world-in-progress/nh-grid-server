import logging
import uvicorn

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    import os
    import sys
    venv_path = sys.prefix
    os.environ['PROJ_LIB'] = os.path.join(venv_path, 'Lib', 'site-packages', 'osgeo', 'data', 'proj')
    
    uvicorn.run("src.nh_grid_server.main:app", host="0.0.0.0", port=8000, reload=True)