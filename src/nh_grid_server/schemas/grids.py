from pydantic import BaseModel

class GridPatch(BaseModel):
    node_key: str  # unique identifier for the patch
    treeger_address: str  # address of the Treeger instance managing this patch

class GridPatches(BaseModel):
    patches: list[GridPatch]  # list of GridPatch instances