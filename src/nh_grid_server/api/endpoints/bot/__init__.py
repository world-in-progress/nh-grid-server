from fastapi import APIRouter
from . import chat

router = APIRouter(prefix='/bot', tags=['bot'])

router.include_router(chat.router)