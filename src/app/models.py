from pydantic import BaseModel
from typing import Optional

class CorrectionRequest(BaseModel):
    """
    Schema for a clinician's feedback on a generated summary.
    """
    original_summary: str
    feedback: str
    report_id: Optional[int] = None # To link feedback to a specific report

class ReportResponse(BaseModel):
    """
    Schema for a single report object returned from the API.
    """
    id: str
    date: str
    category: str
    input_type: str
    trust_score: float
    summary: str

    class Config:
        orm_mode = True
