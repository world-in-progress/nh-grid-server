from fastapi import APIRouter
import c_two as cc
from ...core.config import settings, APP_CONTEXT
from fastapi import APIRouter, Response, HTTPException, Body
import json
import logging
from ...core.bootstrapping_treeger import BT
from ...schemas.project import ResourceCRMStatus
from ...schemas.base import BaseResponse
from pathlib import Path
from icrms.isolution import ISolution, CreateSolutionBody
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/solution', tags=['solution / operation'])
   
@router.get('/{solution_name}', response_model=ResourceCRMStatus)
def check_solution_ready(solution_name: str):
    """
    Description
    --
    Check if the solution runtime resource is ready.
    """
    try:
        node_key = f'root.solutions.{solution_name}'
        tcp_address = BT.instance.get_node_info(node_key).server_address
        flag = cc.rpc.Client.ping(tcp_address)

        return ResourceCRMStatus(
            status='ACTIVATED' if flag else 'DEACTIVATED',
            is_ready=flag
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to check CRM of the solution: {str(e)}')
    
   
@router.post('/create', response_model=BaseResponse)
def create_solution(body: CreateSolutionBody=Body(..., description='create solution')):
    """
    Description
    --
    Create a solution.
    """
    try:
        node_key = f'root.solutions.{body.name}'
        BT.instance.mount_node("solution", node_key, body.model_dump())
        return BaseResponse(
            success=True,
            message=node_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set patch as the current resource: {str(e)}')