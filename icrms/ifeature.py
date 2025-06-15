import json
import c_two as cc
import pyarrow as pa
from typing import Any, List
from src.nh_grid_server.schemas.feature import FeatureProperty, UpdateFeaturePropertyBody

# Define transferables ##################################################

@cc.transferable
class UploadInfo:
    def serialize(file_path: str, file_type: str) -> bytes:
        schema = pa.schema([
            pa.field('file_path', pa.string()),
            pa.field('file_type', pa.string()),
        ])
        data = {
            'file_path': file_path,
            'file_type': file_type
        }
        table = pa.Table.from_pylist([data], schema=schema)
        return cc.message.serialize_from_table(table)

    def deserialize(arrow_bytes: bytes) -> tuple[str, str]:
        row = cc.message.deserialize_to_rows(arrow_bytes)[0]
        return (
            row['file_path'],
            row['file_type'],
        )

@cc.transferable
class UploadedFeatureSaveInfo:
    def serialize(file_path: str, feature_json: dict[str, Any], is_edited: bool) -> bytes:
               
        data = {
            'file_path': file_path,
            'feature_json': feature_json,
            'is_edited': is_edited
        }
        return json.dumps(data).encode('utf-8')

    def deserialize(arrow_bytes: bytes) -> tuple[str, dict[str, Any], bool]:
        data = json.loads(arrow_bytes.tobytes().decode('utf-8'))
        return (
            data['file_path'],
            data['feature_json'],
            data['is_edited']
        )

@cc.transferable
class SaveResult:
    def serialize(info: dict[str, bool | str]) -> bytes:
        return json.dumps(info).encode('utf-8')

    def deserialize(res_bytes: memoryview) -> dict[str, bool | str]:
        res = json.loads(res_bytes.tobytes().decode('utf-8'))
        return res

@cc.transferable   
class FeatureSaveInfo:
    def serialize(feature_name: str, feature_json: dict[str, Any]) -> bytes:
        data = {
            'feature_name': feature_name,
            'feature_json': feature_json
        }
        return json.dumps(data).encode('utf-8')

    def deserialize(res_bytes: memoryview) -> tuple[str, dict[str, Any]]:
        data = json.loads(res_bytes.tobytes().decode('utf-8'))
        return (
            data['feature_name'],
            data['feature_json']
        )

@cc.transferable
class GetFeatureJsonInfo:
    def serialize(feature_name: str) -> bytes:
        data = {
            'feature_name': feature_name
        }
        return json.dumps(data).encode('utf-8')

    def deserialize(res_bytes: memoryview) -> str:
        data = json.loads(res_bytes.tobytes().decode('utf-8'))
        return data['feature_name']

# Define ICRM ###########################################################
@cc.icrm
class IFeature:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
    """
    def upload_feature(self, file_path: str, file_type: str) -> dict[str, bool | str]:
        ...

    def save_uploaded_feature(self, file_path: str, feature_json: dict[str, Any], is_edited: bool) -> dict[str, bool | str]:
        ...

    def save_uploaded_feature(self, file_path: str, feature_json: dict[str, Any], is_edited: bool) -> dict[str, bool | str]:
        ...

    def save_feature(self, feature_property: FeatureProperty, feature_json: dict[str, Any]) -> dict[str, bool | str]:
        ...

    def delete_feature(self, feature_id: str) -> dict[str, bool | str]:
        ...

    def update_feature_property(self, feature_id: str, feature_property: UpdateFeaturePropertyBody) -> dict[str, bool | str]:
        ...

    def get_feature_json(self, feature_name: str) -> dict[str, Any]:
        ...

    def get_feature_meta(self) -> list[FeatureProperty]:
        ...