import json
from typing import Union
from pathlib import Path
from icrms.isolution import ISolution
from ...schemas.base import BaseResponse
from ...core.bootstrapping_treeger import BT
from fastapi import APIRouter, HTTPException, Body
from icrms.itreeger import ReuseAction, CRMDuration

from ...schemas.solution import (
    CreateSolutionBody, ActionType, ActionTypeResponse, ActionTypeDetailResponse,
    AddHumanActionBody, DeleteHumanActionBody, AddFenceParams, TransferWaterParams, AddGateParams, 
    UpdateHumanActionBody, ModelTypeResponse, GetSolutionResponse
)

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/solution', tags=['solution / operation'])

def convert_type_to_frontend(type_annotation) -> str:
    """将Python类型转换为前端可识别的类型"""
    type_str = str(type_annotation)
    
    # 基础类型映射
    type_mapping = {
        'int': 'number',
        'float': 'number', 
        'str': 'string',
        'bool': 'boolean',
        'dict': 'object',
        'list': 'array'
    }
    
    # 处理泛型类型
    if 'dict[str, ' in type_str:
        return 'object'
    elif 'list[' in type_str:
        return 'array'
    elif type_str in type_mapping:
        return type_mapping[type_str]
    elif 'typing.Any' in type_str or 'Any' in type_str:
        return 'any'
    elif 'Union[' in type_str:
        return 'union'
    elif hasattr(type_annotation, '__members__'):
        # 枚举类型
        return 'enum'
    else:
        # 对于其他复杂类型，提取类名
        if hasattr(type_annotation, '__name__'):
            return type_annotation.__name__
        else:
            # 提取最后一个点后的内容作为类型名
            parts = type_str.split('.')
            if parts:
                clean_type = parts[-1].replace('>', '').replace("'", '')
                return clean_type
    
    return 'unknown'    

@router.get('/model_type_list', response_model=ModelTypeResponse)
def get_model_type_list():
    """
    Get all model types from process_group.json.
    """
    try:
        # 获取项目根目录下的 persistence/process_group.json 文件路径
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent  # 回到项目根目录
        process_group_file = project_root / "persistence" / "process_group.json"
        
        if not process_group_file.exists():
            logger.error(f"process_group.json not found at: {process_group_file}")
            return ModelTypeResponse(success=False, data=[])
        
        # 读取并解析 JSON 文件
        with open(process_group_file, 'r', encoding='utf-8') as f:
            process_groups = json.load(f)
        
        model_types = []
        for group in process_groups:
            # 从配置中获取需要跳过的参数，如果没有配置则默认为空集合
            skip_params = set(group.get("skip_parameters", []))
            
            group_data = {
                "group_type": group.get("group_type", ""),
                "description": group.get("description", ""),
                "processes": []
            }
            
            # 处理每个进程的参数信息
            for process in group.get("processes", []):
                process_data = {
                    "name": process.get("name", ""),
                    "parameters": []
                }
                
                # 过滤参数，跳过配置中指定的参数名
                for param in process.get("parameters", []):
                    param_name = param.get("name", "")
                    if param_name not in skip_params:
                        process_data["parameters"].append({
                            "name": param_name,
                            "type": param.get("type", "")
                        })
                
                group_data["processes"].append(process_data)
            
            model_types.append(group_data)
        
        return ModelTypeResponse(success=True, data=model_types)
    except Exception as e:
        logger.error(f'Failed to get model type list: {str(e)}')
        return ModelTypeResponse(success=False, data=[])

@router.get('/action_type_list', response_model=ActionTypeDetailResponse)
def get_action_type_list():
    """
    Description
    --
    Get all action types with complete information including parameter schemas.
    """
    try:
        param_schemas = {
            "add_fence": AddFenceParams,
            "transfer_water": TransferWaterParams,
            "add_gate": AddGateParams
        }
        
        action_types = []
        for action_type in ActionType:
            param_class = param_schemas.get(action_type.action_value)
            param_schema = {}
            
            if param_class:
                param_schema = {
                    "fields": {},
                    "required": []
                }
                
                for field_name, field_info in param_class.model_fields.items():
                    if field_name == "action_type":  # Skip discriminator field
                        continue
                    
                    field_schema = {
                        "required": field_info.is_required()
                    }
                    
                    # Priority 1: Check for enum types (including Optional enums)
                    enum_annotation = None
                    is_optional_enum = False
                    
                    # Direct enum type (e.g., LanduseType)
                    if hasattr(field_info.annotation, '__members__'):
                        enum_annotation = field_info.annotation
                    # Check for Union types (including | syntax and Union[] syntax)
                    else:
                        # Handle Python 3.10+ | syntax (types.UnionType)
                        if str(type(field_info.annotation)) == "<class 'types.UnionType'>":
                            # For types.UnionType, we need to check __args__ directly
                            if hasattr(field_info.annotation, '__args__'):
                                args = field_info.annotation.__args__
                                for arg in args:
                                    if hasattr(arg, '__members__') and arg is not type(None):
                                        enum_annotation = arg
                                        # Check if it's optional (contains None)
                                        if type(None) in args:
                                            is_optional_enum = True
                                        break
                        # Handle traditional Union[] syntax
                        elif hasattr(field_info.annotation, '__origin__'):
                            origin = field_info.annotation.__origin__
                            if origin is Union:
                                args = field_info.annotation.__args__
                                for arg in args:
                                    if hasattr(arg, '__members__') and arg is not type(None):
                                        enum_annotation = arg
                                        # Check if it's optional (contains None)
                                        if type(None) in args:
                                            is_optional_enum = True
                                        break
                    
                    # If enum type is found, set enum-related information
                    if enum_annotation:
                        field_schema["type"] = "enum"
                        
                        # Get actual enum values (for API transmission)
                        enum_items = list(enum_annotation.__members__.items())
                        try:
                            field_schema["enum_options"] = [item.value for _, item in enum_items]
                        except AttributeError:
                            # If enum has no value attribute, use name as option
                            field_schema["enum_options"] = [name for name, _ in enum_items]
                        
                        if is_optional_enum:
                            field_schema["optional"] = True
                    
                    # If not enum type, handle other types
                    else:
                        # Convert type to frontend-recognizable format
                        frontend_type = convert_type_to_frontend(field_info.annotation)
                        field_schema["type"] = frontend_type
                        
                        # Handle Optional types
                        # Check for Python 3.10+ | syntax (types.UnionType)
                        if str(type(field_info.annotation)) == "<class 'types.UnionType'>":
                            if hasattr(field_info.annotation, '__args__'):
                                args = field_info.annotation.__args__
                                if len(args) == 2 and type(None) in args:
                                    # Optional type (e.g., float | None)
                                    non_none_type = args[0] if args[1] is type(None) else args[1]
                                    field_schema["type"] = convert_type_to_frontend(non_none_type)
                                    field_schema["optional"] = True
                                else:
                                    field_schema["type"] = "union"
                                    field_schema["union_types"] = [convert_type_to_frontend(arg) for arg in args]
                        # Handle traditional Union[] syntax
                        elif hasattr(field_info.annotation, '__origin__'):
                            if field_info.annotation.__origin__ is Union:
                                args = field_info.annotation.__args__
                                if len(args) == 2 and type(None) in args:
                                    # Optional type (e.g., Optional[float])
                                    non_none_type = args[0] if args[1] is type(None) else args[1]
                                    field_schema["type"] = convert_type_to_frontend(non_none_type)
                                    field_schema["optional"] = True
                                else:
                                    field_schema["type"] = "union"
                                    field_schema["union_types"] = [convert_type_to_frontend(arg) for arg in args]
                    
                    # Add field description and default value information
                    if hasattr(field_info, 'description') and field_info.description:
                        field_schema["description"] = field_info.description
                    
                    # Handle special frontend-specific information
                    current_type = field_schema.get("type", "")
                    if current_type == "object":
                        field_schema["format"] = "geojson" if field_name == "feature" else "object"
                    elif current_type == "number":
                        if "float" in str(field_info.annotation):
                            field_schema["format"] = "float"
                        else:
                            field_schema["format"] = "integer"
                    
                    param_schema["fields"][field_name] = field_schema
                    
                    if field_info.is_required():
                        param_schema["required"].append(field_name)
            
            action_data = {
                "value": action_type.action_value,
                "name": action_type.display_name,
                "description": action_type.description,
                "param_schema": param_schema
            }
            action_types.append(action_data)
        
        return ActionTypeDetailResponse(
            success=True,
            data=action_types
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get action types: {str(e)}')

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
        BT.instance.mount_node("actions", f'{node_key}.actions')
        BT.instance.mount_node("human_actions", f'{node_key}.actions.human_actions')
        return BaseResponse(
            success=True,
            message=node_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to set patch as the current resource: {str(e)}')
    
@router.post('/add_human_action', response_model=BaseResponse)
def add_human_action(body: AddHumanActionBody=Body(..., description='add human action')):
    """
    Description
    --
    Add a human action.
    """
    try:
        with BT.instance.connect(body.node_key, ISolution, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as solution:
            action_id = solution.add_human_action(body.action_type, body.params)
        BT.instance.mount_node("human_action", f'{body.node_key}.actions.human_actions.{action_id}')
        return BaseResponse(
            success=True,
            message="Action added successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to add human action: {str(e)}')

@router.get('/get_human_actions/{node_key}', response_model=ActionTypeResponse)
def get_human_actions(node_key: str):
    """
    Get human actions for a solution.
    """
    try:
        with BT.instance.connect(node_key, ISolution) as solution:
            actions = solution.get_human_actions()
        return ActionTypeResponse(
            success=True,
            data=actions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get human actions: {str(e)}')

@router.put('/update_human_action', response_model=BaseResponse)
def update_human_action(body: UpdateHumanActionBody=Body(..., description='update human action')):
    """
    Description
    --
    Update a human action.
    """
    try:
        with BT.instance.connect(body.node_key, ISolution) as solution:
            solution.update_human_action(body.action_id, body.params)
        return BaseResponse(
            success=True,
            message="Action updated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to update human action: {str(e)}')

@router.delete('/delete_human_action', response_model=BaseResponse)
def delete_human_action(body: DeleteHumanActionBody=Body(..., description='delete human action')):
    """
    Description
    --
    Delete a human action.
    """
    try:
        with BT.instance.connect(body.node_key, ISolution) as solution:
            solution.delete_human_action(body.action_id)
        BT.instance.unmount_node(f'{body.node_key}.actions.human_actions.{body.action_id}')
        return BaseResponse(
            success=True,
            message="Action deleted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete human action: {str(e)}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to add human action: {str(e)}')
    
@router.get('/package/{node_key}', response_model=BaseResponse)
def package_solution(node_key: str):
    """
    Package a solution.
    """
    try:
        with BT.instance.connect(node_key, ISolution, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as solution:
            result = solution.package()
        return BaseResponse(
            success=result.get('status', False),
            message=result.get('message', 'Solution packaged successfully'),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to package solution: {str(e)}')

@router.get('/{node_key}', response_model=GetSolutionResponse)
def get_solution(node_key: str):
    """
    Get a solution by node key.
    """
    try:
        with BT.instance.connect(node_key, ISolution, duration=CRMDuration.Forever, reuse=ReuseAction.REPLACE) as solution:
            return GetSolutionResponse(
                success=True,
                data=solution.get_solution()
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to get solution: {str(e)}')

@router.delete('/{node_key}', response_model=BaseResponse)
def delete_solution(node_key: str):
    """
    Delete a solution.
    """
    try:
        with BT.instance.connect(node_key, ISolution) as solution:
            solution.delete_solution()
        BT.instance.unmount_node(node_key)
        return BaseResponse(
            success=True,
            message="Solution deleted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to delete solution: {str(e)}')