from pydantic import BaseModel

class PatchNodeInfo(BaseModel):
    node_key: str  # unique identifier for the patch
    treeger_address: str  # address of the Treeger instance managing this patch

class PatchNodeInfos(BaseModel):
    patches: list[PatchNodeInfo]  # list of GridPatch instances