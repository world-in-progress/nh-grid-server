from fastapi import APIRouter
from . import schema
from . import schemas

router = APIRouter(tags=['schema-related apis'])

router.include_router(schema.router)
router.include_router(schemas.router)