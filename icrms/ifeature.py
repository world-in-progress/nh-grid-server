import json
import c_two as cc
import pyarrow as pa
from typing import Any, List

# Define transferables ##################################################

@cc.transferable
class UploadInfo:    
    def serialize(info: dict[str, bool | str]) -> bytes:
        return json.dumps(info).encode('utf-8')

    def deserialize(res_bytes: memoryview) -> dict[str, bool | str]:
        res = json.loads(res_bytes.tobytes().decode('utf-8'))
        return {
            'success': res['success'],
            'file_path': res['file_path'],
            'patch_name': res['patch_name']
        }

@cc.transferable
class SaveInfo:
    """
    Save Info
    ---
    - success (bool): whether the operation is successful
    - message (str): operation result message
    - resource_id (str): the id of the resource in the pool
    - is_edited (bool): whether the feature has been edited
    """
    success: bool
    message: str
    resource_path: str

# Define ICRM ###########################################################

@cc.icrm
class IFeature:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
    """
    def upload_feature(self, file_path: str, file_type: str, patch_name: str) -> UploadInfo:
        ...

    def save_vector_feature(self, feature_json: str, is_edited: bool) -> SaveInfo:
        ...
        
    def get_feature_list_of_patch(self, patch_name: str) -> list[dict[str, Any]]:
        ...
    
    