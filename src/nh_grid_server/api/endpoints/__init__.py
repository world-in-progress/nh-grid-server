from fastapi import APIRouter
from . import schema
from . import feature
from . import scene
from . import patch
from . import topo
from . import proxy
from . import solution

router = APIRouter(prefix='/api', tags=['api'])

router.include_router(schema.router)
router.include_router(patch.router)
router.include_router(topo.router)
router.include_router(feature.router)
router.include_router(scene.router)
router.include_router(proxy.router)
router.include_router(solution.router)
