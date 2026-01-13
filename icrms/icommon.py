import c_two as cc

# Define ICRM ###########################################################

@cc.icrm(namespace='Gridman', version='1.0.0')
class ICommon:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
    """
    def copy_to(self, target_path: str) -> dict:
        ...

    def get_data(self) -> dict:
        ...