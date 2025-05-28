import c_two as cc
from icrms.itree import ITree, MountInfo, TreeNode
import logging
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@cc.iicrm
class Tree(ITree):
    def __init__(self, tree_meta_path: str):
        self.tree_meta_path = tree_meta_path
        if not os.path.exists(tree_meta_path):
            with open(tree_meta_path, 'w') as f:
                yaml.dump([], f)
        logger.info(f'Tree initialized with tree meta path: {tree_meta_path}')

    def mount_tree(self, resource_path: str, patch_name: str, refered_patch: list[str]=[]) -> MountInfo:
        """
        Mount a resource to the tree
        ---
        Args:
            resource_path: path of the resource to mount
            patch_name: name of the patch
            refered_patch: list of patches that this resource refers to
        Returns:
            Mount operation result
        """
        try:
            with open(self.tree_meta_path, 'r') as f:
                meta = yaml.safe_load(f) or []
            
            # 查找或创建patch节点
            patch_node = None
            for node in meta:
                if node['index'] == patch_name:
                    patch_node = node
                    break
            
            if patch_node is None:
                # 如果patch不存在，创建新的patch节点
                patch_node = {
                    'index': patch_name,
                    'children': []
                }
                meta.append(patch_node)
            
            # 在patch下添加资源
            resource_node = {
                'index': os.path.basename(resource_path),
                'path': resource_path,
                'refered_patch': refered_patch,
                'children': []
            }
            patch_node['children'].append(resource_node)
            
            # 保存更新后的meta.yaml
            with open(self.tree_meta_path, 'w') as f:
                yaml.dump(meta, f)
            
            return MountInfo(
                success=True,
                message="Resource mounted successfully",
                path=patch_name + '/' + os.path.basename(resource_path)
            )
            
        except Exception as e:
            logger.error(f'Failed to mount resource: {str(e)}')
            return MountInfo(
                success=False,
                message=str(e),
                path=""
            )

    def unmount_tree(self, node_id: str) -> MountInfo:
        ...
        
    def get_tree(self, node_id: str = None) -> TreeNode:
        ...

    def update_tree(self, node_id: str, new_path: str) -> MountInfo:
        ...