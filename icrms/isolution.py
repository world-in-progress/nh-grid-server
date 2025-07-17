import c_two as cc
from pydantic import BaseModel

class CreateSolutionBody(BaseModel):
    name: str
    env: dict

@cc.icrm
class ISolution:

    def clone_env(self) -> dict:
        """
        克隆环境变量
        :return: 环境变量
        """
        ...

    def get_env(self) -> dict:
        """
        获取环境变量字典
        :return: 环境变量
        """
        ...