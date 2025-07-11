from fastapi import APIRouter
from . import schema
from . import feature
from . import scene
from . import patch

router = APIRouter(prefix='/api', tags=['api'])

router.include_router(schema.router)
router.include_router(patch.router)
router.include_router(feature.router)
router.include_router(scene.router)
