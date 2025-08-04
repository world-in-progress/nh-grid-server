import base64
import c_two as cc
from pathlib import Path
from ...schemas.base import BaseResponse
from icrms.isimulation import ISimulation
from fastapi import APIRouter, HTTPException, Body
from src.nh_grid_server.schemas.simulation import GetStepResultRequest

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/simulation', tags=['simulation / operation'])

@router.post('/step_result', response_model=BaseResponse)
def get_step_result(body: GetStepResultRequest=Body(..., description='get step result')):
    """
    Description
    --
    Get step result and save files.
    """
    try:
        simulation_name = body.simulation_name
        with cc.compo.runtime.connect_crm(body.simulation_address, ISimulation) as simulation:
            step_result = simulation.get_step_result(body.step)
        if step_result is not None:
            # 解析step_result结构
            step = step_result.get('step')
            data = step_result.get('data', {})
            file_types = step_result.get('file_types', [])
            file_suffix = step_result.get('file_suffix', {})
            
            # 创建保存目录
            save_dir = Path('resource') / 'simulations' / simulation_name / str(step)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # 遍历file_types，保存对应的数据
            for file_type in file_types:
                if file_type in data:
                    file_data = data[file_type]
                    suffix = file_suffix.get(file_type, '')
                    filename = f"{file_type}{suffix}"
                    file_path = save_dir / filename
                    
                    try:
                        # 检查file_data是否为字典格式且包含type和content字段
                        if isinstance(file_data, dict) and 'type' in file_data and 'content' in file_data:
                            data_type = file_data.get('type')
                            content = file_data.get('content')
                            
                            if data_type == 'binary':
                                # 处理二进制数据（base64编码）
                                if isinstance(content, str):
                                    binary_data = base64.b64decode(content)
                                    with open(file_path, 'wb') as f:
                                        f.write(binary_data)
                                    logger.info(f"Saved binary file: {file_path}")
                                else:
                                    logger.warning(f"Binary content for {file_type} is not a string")
                                    continue
                            elif data_type == 'text':
                                # 处理文本数据
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    if isinstance(content, list):
                                        # 列表逐行写入
                                        for line in content:
                                            f.write(str(line) + '\n')
                                    elif isinstance(content, dict):
                                        import json
                                        json.dump(content, f, indent=2, ensure_ascii=False)
                                    else:
                                        f.write(str(content))
                                logger.info(f"Saved text file: {file_path}")
                            else:
                                logger.warning(f"Unknown data type '{data_type}' for {file_type}")
                                continue
                        
                        saved_files.append(str(file_path))
                        
                    except Exception as file_error:
                        logger.error(f"Failed to save file {filename}: {str(file_error)}")
                        continue
            
            return BaseResponse(
                success=True,
                message=f"Successfully saved {len(saved_files)} files to {save_dir}. Files: {', '.join([Path(f).name for f in saved_files])}"
            )
        else:
            return BaseResponse(
                success=False,
                message=f"Step {body.step} is not ready or already retrieved."
            )
    except Exception as e:
        logger.error(f"Failed to get step result: {str(e)}")
        raise HTTPException(status_code=500, detail=f'Failed to get step result: {str(e)}')