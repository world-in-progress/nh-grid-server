import json
import c_two as cc
import pyarrow as pa

# Define transferables ##################################################



# Define ICRM ###########################################################

@cc.icrm
class IFeature:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
    """
    def upload_feature(self, feature: dict[str, Any]) -> dict[str, Any]:
        ...
    def save_feature(self, feature: dict[str, Any]) -> dict[str, Any]:
        ...
    def get_feature_list(self, feature: dict[str, Any]) -> dict[str, Any]:
        ...
    
    