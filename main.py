import uvicorn
from src.nh_grid_server.main import app

if __name__ == "__main__":
    uvicorn.run("src.nh_grid_server.main:app", host="0.0.0.0", port=8000, reload=True)