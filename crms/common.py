import os
import shutil
import c_two as cc
from icrms.icommon import ICommon

import logging
logger = logging.getLogger(__name__)

@cc.iicrm
class Common(ICommon):
    """
    CRM
    =
    The Common Resource.  
    Common is a common file that can be uploaded to the resource pool.  
    """
    def __init__(self, name: str, type: str, src_path: str):
        """Method to initialize Common

        Args:
            src_path (str): path to the source file
        """
        self.name = name
        self.type = type
        self.src_path = src_path

    def copy_to(self, target_path: str) -> dict:
        """Method to copy the file to the target path

        Args:
            target_path (str): path to the target file

        Returns:
            dict: result of the copy operation
        """
        try:
            file_name = os.path.basename(self.src_path)
            dst_path = os.path.join(target_path, file_name)
            shutil.copy(self.src_path, dst_path)
            return {"status": True, "message": f"File copied to {dst_path}"}
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return {"status": False, "message": str(e)}