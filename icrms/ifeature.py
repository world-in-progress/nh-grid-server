import c_two as cc
from typing import Any
from src.nh_grid_server.schemas.feature import UpdateFeatureBody

# Define ICRM ###########################################################
@cc.icrm
class IFeature:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
    """
    def save_feature(self, feature_json: dict[str, Any]) -> dict[str, bool | str]:
        ...

    def save_uploaded_feature(self, file_path: str, file_type: str) -> dict[str, bool | str]:
        ...

    def get_feature_json_visualization(self) -> dict[str, Any]:
        ...

    def get_feature_json_computation(self) -> dict[str, Any]:
        ...

    def update_feature(self, update_body: UpdateFeatureBody) -> dict[str, bool | str]:
        ...

    def delete_feature(self) -> dict[str, bool | str]:
        ...
