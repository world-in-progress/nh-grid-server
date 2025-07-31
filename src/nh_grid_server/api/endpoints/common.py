from fastapi import APIRouter
from icrms.icommon import ICommon
from fastapi import APIRouter, Body

from ...schemas.base import BaseResponse
from ...core.bootstrapping_treeger import BT
from crms.treeger import ReuseAction, CRMDuration
from ...schemas.common import CreateCommonBody, CopyToBody, GetDataResponse

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/common', tags=['common / operation'])

@router.post('/create_common', response_model=BaseResponse)
def create_common(body: CreateCommonBody=Body(..., description='create common')):
    """
    Description
    --
    Create a common.
    """
    try:
        node_key = f'root.{body.type}s.{body.name}'
        BT.instance.mount_node(body.type, node_key, body.model_dump())
        return BaseResponse(
            success=True,
            message=node_key
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to create common: {str(e)}'
        )
    
@router.post('/copy_to/{node_key}', response_model=BaseResponse)
def copy_to(node_key: str, body: CopyToBody=Body(..., description='copy common')):
    """
    Description
    --
    Copy a common.
    """
    try:
        with BT.instance.connect(node_key, ICommon, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as common:
            result = common.copy_to(body.target_path)
        return BaseResponse(
            success=result.get("status", False),
            message=result.get("message", "")
        )
    except Exception as e:
        return BaseResponse(
            success=False,
            message=f'Failed to copy common: {str(e)}'
        )
    
@router.get('/get_data/{node_key}', response_model=GetDataResponse)
def get_data(node_key: str):
    """
    Description
    --
    Get data of a common.
    """
    try:
        with BT.instance.connect(node_key, ICommon, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as common:
            data = common.get_data()
        return GetDataResponse(
            success=True,
            message="Data retrieved successfully.",
            data=data
        )
    except Exception as e:
        return GetDataResponse(
            success=False,
            message=f'Failed to get data: {str(e)}',
            data={}
        )