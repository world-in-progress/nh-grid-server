# import json
# import c_two as cc
# import pyarrow as pa
# from typing import Any, List

# # Define transferables ##################################################

# @cc.transferable
# class TreeNode:
#     """
#     Tree Node
#     ---
#     - id (str): node id
#     - name (str): node name
#     - type (str): node type (file/directory)
#     # - path (str): file path
#     """
#     index: str
#     name: str
#     type: str
#     path: str
#     refered_patch: list[str]
    
#     def serialize(data: dict[str, Any]) -> bytes:
#         arrow_schema = pa.schema([
#             pa.field('index', pa.string()),
#             pa.field('name', pa.string()),
#             pa.field('type', pa.string()),
#             pa.field('path', pa.string()),
#             pa.field('refered_patch', pa.list_(pa.string()))
#         ])
        
#         table = pa.Table.from_pylist([data], schema=arrow_schema)
#         return cc.message.serialize_from_table(table)
    
#     def deserialize(res_bytes: memoryview) -> dict[str, Any]:
#         row = cc.message.deserialize_to_rows(res_bytes)[0]
#         return {
#             'index': row['index'],
#             'name': row['name'],
#             'type': row['type'],
#             'path': row['path'],
#             'refered_patch': row['refered_patch']
#         }

# @cc.transferable
# class MountInfo:
#     """
#     Mount Info
#     ---
#     - success (bool): whether the operation is successful
#     - message (str): operation result message
#     - path (str): the path of the mounted node
#     """
#     success: bool
#     message: str
#     path: str
    
#     def serialize(info: dict[str, Any]) -> bytes:
#         arrow_schema = pa.schema([
#             pa.field('success', pa.bool_()),
#             pa.field('message', pa.string()),
#             pa.field('path', pa.string())
#         ])
        
#         table = pa.Table.from_pylist([info], schema=arrow_schema)
#         return cc.message.serialize_from_table(table)
    
#     def deserialize(res_bytes: memoryview) -> dict[str, Any]:
#         row = cc.message.deserialize_to_rows(res_bytes)[0]
#         return {
#             'success': row['success'],
#             'message': row['message'],
#             'path': row['path']
#         }

# # Define ICRM ###########################################################

# @cc.icrm
# class ITree:
#     """
#     ICRM
#     =
#     Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
#     """
#     def mount_tree(self, resource_path: str, patch_name: str, refered_patch: list[str]) -> MountInfo:
#         ...
        
#     def unmount_tree(self, node_index: str) -> MountInfo:
#         ...
        
#     def get_tree(self, node_index: str = None) -> TreeNode:
#         ...
        
#     def update_tree(self, node_index: str, new_path: str) -> MountInfo:
#         ...
    
    