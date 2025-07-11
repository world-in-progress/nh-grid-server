from pydantic import BaseModel, field_validator

class CRMStatus(BaseModel):
    """Status of the project"""
    status: str # 'ACTIVATED', 'DEACTIVATED'
    is_ready: bool # True if the project is ready to be used

    @field_validator('status')
    def validate_status(cls, v):
        if v not in ['ACTIVATED', 'DEACTIVATED']:
            raise ValueError('status must be either "ACTIVATED" or "DEACTIVATED"')
        return v