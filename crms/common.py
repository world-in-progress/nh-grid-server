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
            return {"status": True, "message": file_name}
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return {"status": False, "message": str(e)}
        
    def get_data(self) -> dict:
        """Method to get the data of the Common resource

        Returns:
            dict: data of the Common resource
        """

        data = {}
        if os.path.exists(self.src_path):
            with open(self.src_path, 'r', encoding='utf-8') as file:
                if self.type in ["rainfall", "tide"]:
                    # 对于rainfall类型，按行读取并存入列表
                    data = file.readlines()
                    # 去除每行末尾的换行符
                    data = [line.rstrip('\n') for line in data]
                else:
                    data = file.read()
        else:
            logger.warning(f"Source path {self.src_path} does not exist.")

        return {
            "name": self.name,
            "type": self.type,
            "data": data
        }
    
    def delete(self) -> dict:
        """Method to delete the Common resource file

        Returns:
            dict: result of the delete operation
        """
        try:
            if os.path.exists(self.src_path):
                os.remove(self.src_path)
                logger.info(f"Successfully deleted file: {self.src_path}")
                return {"status": True, "message": f"File {self.name} deleted successfully"}
            else:
                logger.warning(f"File {self.src_path} does not exist")
                return {"status": False, "message": f"File {self.name} does not exist"}
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {"status": False, "message": str(e)}