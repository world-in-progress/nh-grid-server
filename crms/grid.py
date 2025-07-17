import sqlite3
import c_two as cc
from pathlib import Path
from contextlib import contextmanager

from icrms.igrid import IGrid, PatchInfo
from icrms.ipatch import IPatch

@cc.iicrm
class Grid(IGrid):
    def __init__(self, workspace: str):
        self.path = Path(workspace)
        self.db_path = self.path / 'grid.db'
        self.path.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS grid_patches (
                    node_key TEXT,
                    treeger_address TEXT,
                    PRIMARY KEY (node_key, treeger_address)
                )
            """)
            conn.commit()
    
    @contextmanager
    def _connect_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def has_patch(self, patch_info: PatchInfo) -> bool:
        with self._connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM grid_patches 
                WHERE node_key = ? AND treeger_address = ?
            """, (patch_info.node_key, patch_info.treeger_address))
            return cursor.fetchone() is not None

    def add_patch(self, patch_info: PatchInfo):
        if self.has_patch(patch_info):
            return
        
        with self._connect_db() as conn:
            conn.execute("""
                INSERT INTO grid_patches (node_key, treeger_address)
                VALUES (?, ?)
            """, (patch_info.node_key, patch_info.treeger_address))
            conn.commit()

    def remove_patch(self, patch_info: PatchInfo):
        if not self.has_patch(patch_info):
            return
        
        with self._connect_db() as conn:
            conn.execute("""
                DELETE FROM grid_patches 
                WHERE node_key = ? AND treeger_address = ?
            """, (patch_info.node_key, patch_info.treeger_address))
            conn.commit()
    
    def is_not_empty(self) -> bool:
        with self._connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM grid_patches")
            count = cursor.fetchone()[0]
            return count > 0
    
    def list_patches(self) -> list[PatchInfo]:
        with self._connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT node_key, treeger_address FROM grid_patches")
            rows = cursor.fetchall()
            return [PatchInfo(node_key=row['node_key'], treeger_address=row['treeger_address']) for row in rows]
    
    def clear_patches(self):
        with self._connect_db() as conn:
            conn.execute("DELETE FROM grid_patches")
            conn.commit()
    
    def set_patches(self, patch_infos: list[PatchInfo]):
        if self.is_not_empty():
            self.clear_patches()
            
        for patch_info in patch_infos:
            self.add_patch(patch_info)
    
    # def merge(self):