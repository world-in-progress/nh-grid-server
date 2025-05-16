import logging
import uvicorn

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    uvicorn.run("src.nh_grid_server.main:app", host="0.0.0.0", port=8000, reload=True)