import logging
from fastapi import APIRouter
from fastapi import APIRouter, HTTPException

from ...core.bootstrapping_treeger import BT
# from ...schemas.scene import ResponseOfSceneNode
from icrms.itreeger import SceneNodeMeta, ScenarioNodeDescription

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/scene', tags=['scene'])

@router.get('/', response_model=SceneNodeMeta)
def get_scene_node_info(node_key: str, child_start_index: int = 0, child_end_index: int = None):
    """
    Description
    --
    Get information about a specific scene node.
    """
    try:
        if node_key == '_':
            node_key = 'root'
        
        meta = BT.instance.get_scene_node_info(node_key, child_start_index, child_end_index)
        if meta is None:
            raise HTTPException(status_code=404, detail=f'Scene node with key "{node_key}" not found')
        
        return meta
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get scene node info: {str(e)}')

@router.get('/scenario', response_model=list[ScenarioNodeDescription])
def get_scenario_description():
    """
    Description
    --
    Get the scenario description of the current scene.
    """
    try:
        descriptions = BT.instance.get_scenario_description()
        return descriptions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get scenario description: {str(e)}')
