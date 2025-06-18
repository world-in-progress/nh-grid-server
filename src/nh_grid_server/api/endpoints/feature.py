from fastapi import APIRouter
import c_two as cc
from icrms.ifeature import IFeature
from ...core.config import settings, APP_CONTEXT
from fastapi import APIRouter, Response, HTTPException, Body
import json
from ...schemas.feature import UploadBody, FeatureSaveBody, UploadedFeatureSaveBody, GetFeatureJsonInfo, FeatureMeta, UpdateFeaturePropertyBody
import logging
from ...core.bootstrapping_treeger import BT
from ...schemas.project import ResourceCRMStatus
from ...schemas.base import BaseResponse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/feature', tags=['feature / operation'])

@router.get('/', response_model=ResourceCRMStatus)
def check_feature_ready():
    """
    Description
    --
    Check if the feature runtime resource is ready.
    """
    try:
        node_key = f'root/projects/{APP_CONTEXT["current_project"]}/{APP_CONTEXT["current_patch"]}/feature'
        tcp_address = BT.instance.get_node_info(node_key).tcp_address
        flag = cc.message.Client.ping(tcp_address)

        return ResourceCRMStatus(
            status='ACTIVATED' if flag else 'DEACTIVATED',
            is_ready=flag
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to check CRM of the feature: {str(e)}')
    
@router.get('/{project_name}/{patch_name}', response_model=BaseResponse)
def set_patch_feature(project_name: str, patch_name: str):
    """
    Description
    --
    Set a specific patch feature as the current crm server.
    """
    # Check if the patch directory exists
    project_dir = Path(settings.GRID_PROJECT_DIR, project_name)
    patch_dir = project_dir / patch_name
    if not patch_dir.exists():
        raise HTTPException(status_code=404, detail=f'Patch {patch_name} not found')
    
    try:
        node_key = f'root/projects/{project_name}/{patch_name}/feature'
        APP_CONTEXT['current_project'] = project_name
        APP_CONTEXT['current_patch'] = patch_name
        BT.instance.activate_node(node_key)
        return BaseResponse(
            success=True,
            message=f'Feature node ({node_key}) activated'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set patch as the current resource: {str(e)}')

@router.get('/meta', response_model=FeatureMeta)
def get_current_feature_meta():
    """
    Get grid meta information of the current patch
    """
    try:
        with BT.instance.connect(_get_current_feature_node(), IFeature) as feature:
            feature_meta = feature.get_feature_meta()
        return FeatureMeta(
            feature_meta=feature_meta
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f'Failed to read project meta file: {str(e)}')

@router.post('/upload', response_description='Returns upload information in json')
def upload_feature(body: UploadBody=Body(..., description='upload feature info')):
    try:
        with BT.instance.connect(_get_current_feature_node(), IFeature) as feature:
            logger.info(f'Uploading feature: {body.file_path} {body.file_type}')
            upload_info = feature.upload_feature(body.file_path, body.file_type)

            logger.info(f'Uploading feature info: {upload_info}')
            
        return Response(
            content=json.dumps(upload_info),
            media_type='application/json'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to upload feature: {str(e)}')

@router.post('/save')
def save_feature(body: FeatureSaveBody=Body(..., description='save feature info')):
    try:
        with BT.instance.connect(_get_current_feature_node(), IFeature) as feature:
            logger.info(f'Saving feature: {body.feature_property.id}')
            save_info = feature.save_feature(body.feature_property, body.feature_json)
        return Response(
            content=json.dumps(save_info),
                media_type='application/json'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save feature: {str(e)}')
    
@router.delete('/{feature_id}')
def delete_feature(feature_id: str):
    try:
        with BT.instance.connect(_get_current_feature_node(), IFeature) as feature:
            delete_info = feature.delete_feature(feature_id)
        return Response(
            content=json.dumps(delete_info),
                media_type='application/json'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete feature: {str(e)}')

@router.put('/{feature_id}')
def update_feature_property(feature_id: str, body: UpdateFeaturePropertyBody=Body(..., description='update feature property')):
    try:
        with BT.instance.connect(_get_current_feature_node(), IFeature) as feature:
            update_info = feature.update_feature_property(feature_id, body)
        return Response(
            content=json.dumps(update_info),
                media_type='application/json'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update feature property: {str(e)}')

@router.post('/save_uploaded')
def save_uploaded_feature(body: UploadedFeatureSaveBody=Body(..., description='save uploaded feature info')):
    try:
        with BT.instance.connect(_get_current_feature_node(), IFeature) as feature:
            save_info = feature.save_uploaded_feature(body.file_path, body.feature_json, body.is_edited)
        return Response(
            content=json.dumps(save_info),
                media_type='application/json'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to save uploaded feature: {str(e)}')
    
@router.post('/get_feature_json')
def get_feature_json(body: GetFeatureJsonInfo=Body(..., description='get feature json info')):
    try:
        with BT.instance.connect(_get_current_feature_node(), IFeature) as feature:
            feature_json = feature.get_feature_json(body.feature_name)
        return Response(
            content=json.dumps(feature_json),
                media_type='application/json'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get feature json: {str(e)}')

# Helpers ##################################################

def _get_current_feature_node():
    return f'root/projects/{APP_CONTEXT.get("current_project")}/{APP_CONTEXT.get("current_patch")}/feature'