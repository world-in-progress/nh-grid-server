import logging
from fastapi import APIRouter
from icrms.ifeature import IFeature
from fastapi import APIRouter, Body

from icrms.itreeger import ReuseAction
from ...schemas.base import BaseResponse
from ...core.bootstrapping_treeger import BT
from crms.treeger import ReuseAction, CRMDuration
from ...schemas.feature import GetFeatureJsonResponse, GetFeatureResponse, UploadFeatureSaveBody, FeatureSaveBody, UpdateFeatureBody, CreateFeatureBody

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/feature', tags=['feature / operation'])
 
@router.post('/create', response_model=BaseResponse)
def create_feature(body: CreateFeatureBody=Body(..., description='create feature')):
    """
    Description
    --
    Create a feature.
    """
    try:
        node_key = f'root.vectors.{body.name}'
        BT.instance.mount_node("vector", node_key, body.model_dump())
        return BaseResponse(
            success=True,
            message=node_key
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to create feature: {str(e)}'
        )

@router.post('/save', response_model=BaseResponse)
def save_feature(body: FeatureSaveBody=Body(..., description='save feature')):
    """
    Description
    --
    Save a feature.
    """
    try:
        with BT.instance.connect(body.node_key, IFeature, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as feature:
            save_info = feature.save_feature(body.feature_json)
        return BaseResponse(
            success=save_info.get('success', False),
            message=save_info.get('message', '')
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to save feature: {str(e)}'
        )

@router.post('/save_uploaded', response_model=BaseResponse)
def save_uploaded_feature(body: UploadFeatureSaveBody=Body(..., description='save uploaded feature')):
    """
    Description
    --
    Save an uploaded feature.
    """
    try:
        with BT.instance.connect(body.node_key, IFeature, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as feature:
            save_info = feature.save_uploaded_feature(body.file_path, body.file_type)
        return BaseResponse(
            success=save_info.get('success', False),
            message=save_info.get('message', '')
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to save uploaded feature: {str(e)}'
        )

@router.get('/{node_key}', response_model=GetFeatureResponse)
def get_feature(node_key: str):
    try:
        with BT.instance.connect(node_key, IFeature, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as feature:
            data = feature.get_feature()
        return GetFeatureResponse(
            success=True,
            message='Feature JSON retrieved successfully',
            data=data
        )
    except Exception as e:
        return GetFeatureResponse(
            success=False,
            message=f'Failed to retrieve feature JSON: {str(e)}',
            data=None
        )

@router.get('/feature_json_computation/{node_key}', response_model=GetFeatureJsonResponse)
def get_feature_json_computation(node_key: str):
    try:
        with BT.instance.connect(node_key, IFeature, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as feature:
            feature_json = feature.get_feature_json_computation()
        return GetFeatureJsonResponse(
            success=True,
            message='Feature JSON retrieved successfully',
            feature_json=feature_json
        )
    except Exception as e:
        return GetFeatureJsonResponse(
            success=False,
            message=f'Failed to retrieve feature JSON: {str(e)}',
            feature_json=None
        )

@router.put('/{node_key}', response_model=BaseResponse)
def update_feature(node_key: str, body: UpdateFeatureBody=Body(..., description='update feature properties')):
    """
    Description
    --
    Update a feature's properties.
    """
    try:
        with BT.instance.connect(node_key, IFeature, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as feature:
            update_info = feature.update_feature(body)
        return BaseResponse(
            success=update_info.get('success', False),
            message=update_info.get('message', '')
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to update feature: {str(e)}'
        )

@router.delete('/{node_key}', response_model=BaseResponse)
def delete_feature(node_key: str):
    """
    Description
    --
    Delete a feature.
    """
    try:
        with BT.instance.connect(node_key, IFeature, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as feature:
            delete_info = feature.delete_feature()
        if delete_info.get('success', False):
            BT.instance.unmount_node(node_key)
        return BaseResponse(
            success=delete_info.get('success', False),
            message=delete_info.get('message', '')
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to delete feature: {str(e)}'
        )
    