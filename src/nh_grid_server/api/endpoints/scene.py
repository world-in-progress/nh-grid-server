import logging
from fastapi import APIRouter
from fastapi import APIRouter, HTTPException

from ...core.bootstrapping_treeger import BT
# from ...schemas.scene import ResponseOfSceneNode
from icrms.itreeger import SceneNodeMeta

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/scene', tags=['scene'])

@router.get('/', response_model=SceneNodeMeta)
def get_scene_node_info(node_key: str):
    """
    Description
    --
    Get information about a specific scene node.
    """
    try:
        if node_key is '':
            node_key = 'root'
        
        meta = BT.instance.get_scene_node_info(node_key)
        if meta is None:
            raise HTTPException(status_code=404, detail=f'Scene node with key "{node_key}" not found')
        
        return SceneNodeMeta(
            node_name=meta.node_name,
            node_degree=meta.node_degree,
            children=meta.children
        ) 
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get scene node info: {str(e)}')
