import c_two as cc

@cc.icrm
class ISolution:

    def get_action_types(self) -> list[str]:
        """
        获取动作类型列表
        :return: 动作类型列表
        """
        ...

    def add_human_action(self, action_type: str, params: dict) -> str:
        """
        添加人工操作
        :param action: 人工操作参数
        :return: 添加结果
        """
        ...

    def update_human_action(self, action_id: str, params: dict) -> None:
        """
        更新人工操作
        :param action_id: 人工操作ID
        :param params: 人工操作参数
        :return: None
        """
        ...

    def delete_human_action(self, action_id: str) -> None:
        """
        删除人工操作
        :param action_id: 人工操作ID
        :return: None
        """
        ...

    def get_human_actions(self) -> list[dict]:
        """
        获取所有人工操作
        :return: 人工操作列表
        """
        ...

    def package(self) -> dict:
        """
        打包解决方案
        :return: 打包结果
        """
        ...

    def delete_solution(self) -> None:
        """
        删除解决方案
        :return: None
        """
        ...

    def get_solution(self) -> dict:
        """
        获取解决方案
        :return: 解决方案字典
        """
        ...

    # From Model Server
    def clone_package(self) -> dict:
        """
        克隆解决方案包
        :return: 解决方案包
        """
        ...

    def get_model_env(self) -> dict:
        """
        获取模型环境字典
        :return: 模型环境字典
        """
        ...

    def get_terrain_data(self) -> dict:
        """
        获取地形数据字典
        :return: 地形数据字典
        """
        ...