from fastapi import APIRouter
from . import schema
from . import feature
from . import scene
from . import patch
from . import topo

router = APIRouter(prefix='/api', tags=['api'])

router.include_router(schema.router)
router.include_router(patch.router)
router.include_router(topo.router)
router.include_router(feature.router)
router.include_router(scene.router)
