import os
import random
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import mongoengine as me
import google.generativeai as genai
from dotenv import load_dotenv

from models import CorrectionRequest, ReportResponse, DownloadRequest
from database import Report
import llm_services
import utils

# --- Load Environment Variables ---
# This will load the variables from the .env file
load_dotenv()

# --- App Initialization ---
app = FastAPI(title="Clinical Diagnosis AI Suite")

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "null", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database & GenAI Client Configuration on Startup ---
@app.on_event("startup")
def startup_event():
    # Configure Google API Key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=google_api_key)
    print("Google Generative AI client configured.")

    # Configure and connect to MongoDB Atlas
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable not set.")
    try:
        me.connect(host=mongo_uri)
        print("Successfully connected to MongoDB Atlas.")
    except Exception as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")

# --- Database Disconnection on Shutdown ---
@app.on_event("shutdown")
def shutdown_db_client():
    me.disconnect()
    print("Disconnected from MongoDB.")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = './ui/index.html'
    with open(html_file, 'r') as file:
        html_content = file.read()  # In a real deployment, consider using StaticFiles to serve static assets
 
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_root():
    return {"status": "Clinical Diagnosis API is running."}

@app.post("/diagnose")
async def diagnose(
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """
    Generates a preliminary diagnostic summary from an image, text, or both.
    This summary is NOT saved to the database until feedback is provided.
    """
    if not image and not text:
        raise HTTPException(status_code=400, detail="Please provide either an image or text.")

    image_data = await image.read() if image else None
    
    try:
        summary = llm_services.generate_summary(image_data=image_data, text_data=text)
        return JSONResponse(content={"summary": {"output": summary}})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


@app.post("/corrections")
async def process_corrections(request: CorrectionRequest):
    """
    Receives feedback, calculates a trust score, regenerates the summary,
    saves the final report to the database, and returns the new summary.
    """
    try:
        trust_score, regenerated_summary = llm_services.calculate_trust_score_and_regenerate(
            original_summary=request.original_summary,
            feedback=request.feedback
        )

        # Create and save the report to MongoDB
        new_report = Report(
            category=random.choice(['X-Ray', 'MRI', 'General', 'Lab']),
            input_type='Image' if 'Image' in request.original_summary else 'Text',
            trust_score=trust_score,
            summary=regenerated_summary,
            details=request.original_summary
        )
        new_report.save()

        return JSONResponse(content={"summary": {"output": regenerated_summary}})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process corrections: {e}")


@app.post("/diagnose/download")
async def download_report(request: DownloadRequest):
    """
    Converts a given summary text into a downloadable PDF file.
    """
    try:
        pdf_bytes = utils.create_pdf_from_summary(request.summary)
        return StreamingResponse(iter([pdf_bytes]), media_type="application/pdf", headers={
            "Content-Disposition": "attachment; filename=diagnosis_summary.pdf"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create PDF: {e}")


@app.get("/reports", response_model=List[ReportResponse])
async def get_reports():
    """
    Fetches all saved reports from the database for the Reporting and
    Evaluations pages.
    """
    try:
        reports = Report.objects().order_by('-generated_date')
        return [report.to_dict() for report in reports]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reports: {e}")

# To run this application:
# 1. Create a .env file with your MONGO_URI and GOOGLE_API_KEY.
# 2. Run `uvicorn app:app --reload` in your terminal.
