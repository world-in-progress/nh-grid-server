import c_two as cc

# Define ICRM ###########################################################
@cc.icrm
class ICommon:
    """
    ICRM
    =
    Interface of Core Resource Model (ICRM) specifies how to interact with CRM. 
    """
    def copy_to(self, target_path: str) -> dict:
        ...