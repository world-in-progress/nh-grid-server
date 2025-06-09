from fastapi import APIRouter
from .endpoints import ui, bot, router

api_router = APIRouter()

api_router.include_router(ui.router)
api_router.include_router(bot.router)
api_router.include_router(router)