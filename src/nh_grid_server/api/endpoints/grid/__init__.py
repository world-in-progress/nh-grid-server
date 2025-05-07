from fastapi import APIRouter
from . import schema
from . import schemas
from . import project
from . import projects
from . import operation
from . import subproject
from . import subprojects

router = APIRouter(prefix='/grid', tags=['grid'])

router.include_router(schema.router)
router.include_router(schemas.router)
router.include_router(project.router)
router.include_router(projects.router)
router.include_router(operation.router)
router.include_router(subproject.router)
router.include_router(subprojects.router)