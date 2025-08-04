from pydantic import BaseModel

class GetStepResultRequest(BaseModel):
    simulation_name: str
    simulation_address: str
    step: int