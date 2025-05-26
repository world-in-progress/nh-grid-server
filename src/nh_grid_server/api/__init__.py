from fastapi import APIRouter
from .endpoints import ui, grid, bot

api_router = APIRouter()

api_router.include_router(ui.router)
api_router.include_router(bot.router)
api_router.include_router(grid.router)