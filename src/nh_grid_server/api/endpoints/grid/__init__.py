from fastapi import APIRouter
from . import schema
from . import schemas
from . import project
from . import projects
from . import operation
from . import patch
from . import patches

router = APIRouter(prefix='/grid', tags=['grid'])

router.include_router(schema.router)
router.include_router(schemas.router)
router.include_router(project.router)
router.include_router(projects.router)

router.include_router(patch.router)
router.include_router(patches.router)

router.include_router(operation.router)
