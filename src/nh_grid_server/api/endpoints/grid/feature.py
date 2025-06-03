from fastapi import APIRouter
import c_two as cc
from icrms.ifeature import IFeature
from ....core.config import settings
from fastapi import APIRouter, Response, HTTPException, Body
import json
from ....schemas.feature import UploadBody, FeatureSaveBody, UploadedFeatureSaveBody
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/feature', tags=['grid / feature'])

@router.post('/upload', response_description='Returns upload information in json')
def upload_feature(body: UploadBody=Body(..., description='upload feature info')):
    with cc.compo.runtime.connect_crm(settings.FEATURE_TCP_ADDRESS, IFeature) as feature_interface:

        logger.info(f'Uploading feature: {body.file_path} {body.file_type} {body.feature_type}')

        upload_info = feature_interface.upload_feature(body.file_path, body.file_type, body.feature_type)

        logger.info(f'Uploading feature info: {upload_info}')
        
        return Response(
            content=json.dumps(upload_info),
            media_type='application/json'
        )

@router.post('/save')
def save_feature(body: FeatureSaveBody=Body(..., description='save feature info')):
    with cc.compo.runtime.connect_crm(settings.FEATURE_TCP_ADDRESS, IFeature) as feature_interface:
        save_info = feature_interface.save_feature(body.feature_name, body.feature_type, body.feature_json)
        return Response(
            content=json.dumps(save_info),
            media_type='application/json'
        )
    
@router.post('/save_uploaded')
def save_uploaded_feature(body: UploadedFeatureSaveBody=Body(..., description='save uploaded feature info')):
    with cc.compo.runtime.connect_crm(settings.FEATURE_TCP_ADDRESS, IFeature) as feature_interface:
        save_info = feature_interface.save_uploaded_feature(body.file_path, body.feature_type, body.feature_json, body.is_edited)
        return Response(
            content=json.dumps(save_info),
            media_type='application/json'
        )
