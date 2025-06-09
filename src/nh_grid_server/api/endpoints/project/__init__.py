from fastapi import APIRouter
from . import project
from . import projects

router = APIRouter(tags=['project-related apis'])

router.include_router(project.router)
router.include_router(projects.router)