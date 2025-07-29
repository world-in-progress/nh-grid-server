from fastapi import APIRouter
from . import schema
from . import feature
from . import scene
from . import patch
from . import topo
from . import proxy
from . import grids
from . import solution
from . import raster
from . import common

router = APIRouter(prefix='/api', tags=['api'])

router.include_router(schema.router)
router.include_router(patch.router)
router.include_router(topo.router)
router.include_router(feature.router)
router.include_router(scene.router)
router.include_router(grids.router)
router.include_router(proxy.router)
router.include_router(solution.router)
router.include_router(raster.router)
router.include_router(common.router)
