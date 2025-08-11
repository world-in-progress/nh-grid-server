import base64
import c_two as cc
import json
import os
from pathlib import Path
from ...schemas.base import BaseResponse
from icrms.isimulation import ISimulation
from fastapi import APIRouter, HTTPException, Body, Request
from src.nh_grid_server.schemas.simulation import GetStepResultRequest, WaterDataResponse
from ...core.bootstrapping_treeger import BT
from ...core.config import settings

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
            # Parse step_result structure
            step = step_result.get('step')
            data = step_result.get('data', {})
            file_types = step_result.get('file_types', [])
            file_suffix = step_result.get('file_suffix', {})
            
            # Create save directory
            save_dir = Path('resource') / 'simulations' / simulation_name / str(step)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # Iterate through file_types and save corresponding data
            for file_type in file_types:
                if file_type in data:
                    file_data = data[file_type]
                    suffix = file_suffix.get(file_type, '')
                    filename = f"{file_type}{suffix}"
                    file_path = save_dir / filename
                    
                    try:
                        # Check if file_data is in dictionary format and contains type and content fields
                        if isinstance(file_data, dict) and 'type' in file_data and 'content' in file_data:
                            data_type = file_data.get('type')
                            content = file_data.get('content')
                            
                            if data_type == 'binary':
                                # Handle binary data (base64 encoded)
                                if isinstance(content, str):
                                    binary_data = base64.b64decode(content)
                                    with open(file_path, 'wb') as f:
                                        f.write(binary_data)
                                    logger.info(f"Saved binary file: {file_path}")
                                else:
                                    logger.warning(f"Binary content for {file_type} is not a string")
                                    continue
                            elif data_type == 'text':
                                # Handle text data
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    if isinstance(content, list):
                                        # Write list line by line
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

@router.get('/get_water_data/{simulation_name}/{step}', response_model=WaterDataResponse)
def get_water_data(simulation_name: str, step: int, request: Request):
    """
    Get water simulation data for a specific step.
    """
    try:     
        simulation_dir = Path(settings.ROOT_DIR) / "resource" / "simulations" / simulation_name
        
        if not simulation_dir.exists():
            raise HTTPException(status_code=404, detail=f'Simulation directory not found: {simulation_name}')
        
        # Build directory path for specific step
        step_dir = simulation_dir / str(step)
        
        if not step_dir.exists():
            raise HTTPException(status_code=404, detail=f'Step {step} not found for simulation: {simulation_name}')
        
        # Read data file for current step
        data_file = step_dir / "data.json"
        
        if not data_file.exists():
            raise HTTPException(status_code=404, detail=f'Data file not found for step {step}')
        
        with open(data_file, 'r', encoding='utf-8') as f:
            step_data = json.load(f)
        
        # Build complete URL for current step's HUV image
        base_url = f"{request.url.scheme}://{request.url.netloc}"
        huv_url = f"{base_url}/simulations/{simulation_name}/{step}/huv.png"
        
        # Extract statistical data
        huv_stats = step_data.get('huv_stats', {})
        
        # Extract latitude and longitude boundaries and calculate center point
        bounds_4326 = step_data.get('bounds_4326', {})
        upper_left = bounds_4326.get('upper_left', {})
        upper_right = bounds_4326.get('upper_right', {})
        lower_left = bounds_4326.get('lower_left', {})
        lower_right = bounds_4326.get('lower_right', {})
        
        upper_left_lon = upper_left.get('lon', 0.0)
        upper_left_lat = upper_left.get('lat', 0.0)
        upper_right_lon = upper_right.get('lon', 0.0)
        upper_right_lat = upper_right.get('lat', 0.0)
        lower_left_lon = lower_left.get('lon', 0.0)
        lower_left_lat = lower_left.get('lat', 0.0)
        lower_right_lon = lower_right.get('lon', 0.0)
        lower_right_lat = lower_right.get('lat', 0.0)
        upper_left = [upper_left_lon, upper_left_lat]
        upper_right = [upper_right_lon, upper_right_lat]
        lower_left = [lower_left_lon, lower_left_lat]
        lower_right = [lower_right_lon, lower_right_lat]
        
        # Depth data
        depth_stats = huv_stats.get('depth', {})
        water_height_min = depth_stats.get('min_value', 0.0)
        water_height_max = depth_stats.get('max_value', 0.0)
        
        # U velocity data
        u_stats = huv_stats.get('u', {})
        velocity_u_min = u_stats.get('min_value', 0.0)
        velocity_u_max = u_stats.get('max_value', 0.0)
        
        # V velocity data
        v_stats = huv_stats.get('v', {})
        velocity_v_min = v_stats.get('min_value', 0.0)
        velocity_v_max = v_stats.get('max_value', 0.0)
        
        # Get map dimensions
        dimensions = step_data.get('dimensions', {})
        water_map_size = [dimensions.get('width', 0), dimensions.get('height', 0)]
        
        # Build return data
        water_data = {
            "durationTime": 2000,
            "waterHuvMaps": huv_url,
            "waterHuvMapsSize": water_map_size,
            "waterHeightMin": water_height_min,
            "waterHeightMax": water_height_max,
            "velocityUMin": velocity_u_min,
            "velocityUMax": velocity_u_max,
            "velocityVMin": velocity_v_min,
            "velocityVMax": velocity_v_max,
            "lower_left": lower_left,
            "lower_right": lower_right,
            "upper_right": upper_right,
            "upper_left": upper_left,
        }
            
        return WaterDataResponse(
            success=True,
            data=water_data
        )
    except Exception as e:
        logger.error(f'Failed to get water data: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to get water data: {str(e)}')
