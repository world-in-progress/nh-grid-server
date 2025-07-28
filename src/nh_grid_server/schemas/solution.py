from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Union, Any, Literal

class ActionType(Enum):
    ADD_FENCE = ("add_fence", "添加基围", "绘制区域以添加基围结构修改网格属性")
    TRANSFER_WATER = ("transfer_water", "调水", "指定两个网格进行水量调配")
    ADD_GATE = ("add_gate", "添加闸门", "绘制区域以添加闸门结构并指定闸门高度和上下游网格")
    
    def __init__(self, action_value, display_name, description):
        self.action_value = action_value
        self.display_name = display_name
        self.description = description
    
    def __str__(self):
        return self.action_value

class ActionTypeResponse(BaseModel):
    success: bool
    data: list[dict]

class ModelTypeResponse(BaseModel):
    success: bool
    data: list[dict]

class ActionTypeDetailResponse(BaseModel):
    success: bool
    data: list[dict]  # 包含更详细的action类型信息，包括参数schema

class CreateSolutionBody(BaseModel):
    name: str
    env: dict
    action_types: list[str]    

class LanduseType(str, Enum):
    POND = "pond"
    FENCE = "fence"
    DRAIN = "drain"
    DAM = "dam"

class AddFenceParams(BaseModel):
    action_type: Literal["add_fence"] = Field(default="add_fence", description="参数类型标识")
    elevation_delta: float | None = None
    landuse_type: LanduseType | None = None
    feature: dict[str, Any]

class TransferWaterParams(BaseModel):
    action_type: Literal["transfer_water"] = Field(default="transfer_water", description="参数类型标识")
    from_grid: int
    to_grid: int
    q: float  # 通量

class AddGateParams(BaseModel):
    action_type: Literal["add_gate"] = Field(default="add_gate", description="参数类型标识")
    ud_stream: int
    gate_height: int
    feature: dict[str, Any]

class AddHumanActionBody(BaseModel):
    solution_name: str
    action_type: str
    params: Union[
        AddFenceParams, 
        TransferWaterParams, 
        AddGateParams
    ] = Field(discriminator='action_type')
    
    @model_validator(mode='before')
    @classmethod
    def set_params_action_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 如果params中没有action_type，使用外层的action_type
            if 'params' in data and isinstance(data['params'], dict):
                if 'action_type' not in data['params']:
                    data['params']['action_type'] = data.get('action_type')
        return data

class UpdateHumanActionBody(BaseModel):
    solution_name: str
    action_id: str
    action_type: str
    params: Union[
        AddFenceParams, 
        TransferWaterParams, 
        AddGateParams
    ] = Field(discriminator='action_type')

    @model_validator(mode='before')
    @classmethod
    def set_params_action_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 如果params中没有action_type，使用外层的action_type
            if 'params' in data and isinstance(data['params'], dict):
                if 'action_type' not in data['params']:
                    data['params']['action_type'] = data.get('action_type')
        return data

class DeleteHumanActionBody(BaseModel):
    solution_name: str
    action_id: str