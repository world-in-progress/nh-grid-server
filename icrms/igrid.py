import c_two as cc
from dataclasses import dataclass

@dataclass
class PatchInfo:
    node_key: str
    treeger_address: str
    
@cc.icrm
class IGrid:
    def add_patch(self, patch_info: PatchInfo):
        ...