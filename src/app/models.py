from pydantic import BaseModel
from typing import Optional

class CorrectionRequest(BaseModel):
    """
    Schema for a clinician's feedback on a generated summary.
    """
    original_summary: str
    feedback: str
    report_id: Optional[int] = None # To link feedback to a specific report

class AcceptReportRequest(BaseModel):
    """
    Schema for a clinician's feedback on a generated summary.
    """
    summary: str

class ReportResponse(BaseModel):
    """
    Schema for a single report object returned from the API.
    """
    id: str
    date: str
    category: str
    inputType: str
    trustScore: float
    summary: str

class ReportModel(BaseModel):
    """
    Schema for a single report object returned from the API.
    """
    id: str
    date: str
    category: str
    input_type: str
    trust_score: float
    summary: str
    details: str

    class Config:
        orm_mode = True

class DownloadRequest(BaseModel):
    """
    Schema for downloading a report.
    """
    summary: str



# End of src/app/models.py