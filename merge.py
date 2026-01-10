import shutil
from pathlib import Path

from crms.assembly import assembly
from crms.hydro import HydroElements

if __name__ == '__main__':
    workspace = Path('./test_grid')
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    
    assembly(
        schema_file_path='./resource/topo/schemas/hk/schema.json',
        patch_paths=[
            './resource/topo/schemas/hk/patches/1'
        ],
        workspace=str(workspace)
    )
    
    es = HydroElements(str(workspace / 'grid_records.bin'))
    es.export_ne(str(workspace / 'ne.txt'))