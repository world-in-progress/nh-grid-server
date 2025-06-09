from fastapi import APIRouter
from . import patch
from . import patches

router = APIRouter(tags=['patch-related apis'])

router.include_router(patch.router)
router.include_router(patches.router)