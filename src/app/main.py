import os
import random
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import mongoengine as me
import google.generativeai as genai
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from models import (
    CorrectionRequest,
    ReportResponse,
    DownloadRequest,
    AcceptReportRequest,
)
from database import Report
import llm_services
import utils
from pydantic import BaseModel
from typing import Dict, Any

from agents.fetch import summarize, load_metadata_jsonl, startup_load
from agents.gemma import load_pipeline_at_startup, infer, convert_images_to_base64

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

# releavtive path to embeddings
embeddings_path = "../embeddings_25k/"
index_path = os.path.join(embeddings_path, "images.index")
meta_path = os.path.join(embeddings_path, "metadata.jsonl")

global METADATA_HISTORY


# Use FastAPI lifespan instead of deprecated startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    # Configure Google API Key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=google_api_key)
    print("Google Generative AI client configured.")

    print("Preparing the Medgemma model...  This may take a while.")
    load_pipeline_at_startup()
    print("Medgemma model is ready.")

    print("Loading FAISS index and metadata...")
    startup_load()
    print("FAISS index and metadata loaded.")

    # Prefer a module-relative absolute path so the server can be started from anywhere
    metadata_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "synthetic_ehr",
            "out_all_patients_context.json",
        )
    )

    # store loaded metadata on the app.state so route handlers can access it reliably
    try:
        app.state.METADATA_HISTORY = utils.load_json_file(metadata_path)
    except Exception:
        # fallback to empty dict to avoid attribute errors in routes
        app.state.METADATA_HISTORY = {}

    # Configure and connect to MongoDB Atlas
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable not set.")
    try:
        me.connect(host=mongo_uri)
        print("Successfully connected to MongoDB Atlas.")
    except Exception as e:
        print(f"Failed to connect to MongoDB Atlas: {e}")

    try:
        yield
    finally:
        # SHUTDOWN
        try:
            me.disconnect()
            print("Disconnected from MongoDB.")
        except Exception:
            pass


app = FastAPI(title="Clinical Diagnosis AI Suite", lifespan=lifespan)

# Mount the 'static' folder at URL path '/static'
statis_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=statis_path), name="static")


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    try:
        with open(html_file, "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read HTML file: {e}")


@app.get("/health")
def health_root():
    return {"status": "Clinical Diagnosis API is running."}


@app.post("/diagnose")
async def diagnose(
    image: Optional[UploadFile] = File(...),
    text: Optional[str] = Form(...),
    is_base_retrival: Optional[bool] = Form(False),
    patient_id: Optional[str] = Form(...),
    corrections: Optional[str] = Form(None),
    previous_summary: Optional[str] = Form(None),
):
    """
    Generates a preliminary diagnostic summary from an image, text, or both.
    This summary is NOT saved to the database until feedback is provided.
    """
    if not image and not text:
        raise HTTPException(
            status_code=400, detail="Please provide either an image or text."
        )

    image_data = await image.read() if image else None
    patient_context = None

    print("Starting diagnosis process...")
    print(
        f"Received parameters - is_base_retrival: {is_base_retrival}, patient_id: {patient_id}, corrections: {corrections is not None}, previous_summary: {previous_summary is not None}"
    )
    print(
        f"Image data size: {len(image_data) if image_data else 'No image provided'}, Text data size: {len(text) if text else 'No text provided'} "
    )

    # return JSONResponse(content={"summary": {"output": "Debugging - process halted"}})

    try:
        if patient_id:
            metadata_store = getattr(app.state, "METADATA_HISTORY", {})
            patient_context = metadata_store.get(
                patient_id, "No relevant patient history found."
            )
            if patient_context is None:
                print(f"No metadata found for patient ID: {patient_id}")
            else:
                print(f"Patient context of the patient is : {patient_context}")
                print(f"Loaded metadata for patient ID: {patient_id}")
        prepare_context = summarize(
            file=image_data,
            is_base_retrival=is_base_retrival,
            patient_id=patient_id,
            patient_context=patient_context,
            corrections=corrections,
            previous_summary=previous_summary,
            text=text
        )
        result = None
        print("Prepared context for inference." + type(prepare_context).__name__)
        result, file = infer(image_data, prepare_context)
        summary = result  # .generated_text
        print("Type of Summary:", type(summary))
        print("Generated Summary:", summary)
        print("Replacing images in the summary...")
        print("File data for image conversion:", file)
        summary = convert_images_to_base64(summary)
        # print("Final Summary after replacing images:", summary)
        print("Diagnosis process completed.")
        # import json
        # file = "C:\\Users\\lokesh-g\\Desktop\\sudss-app\\inference_output_20251020_160742.json"
        # with open(file, "r", encoding='utf-8') as f:
        #     summary = f.read()
        #     summary = json.loads(summary)
        # summary = llm_services.generate_summary(image_data=image_data, text_data=text)
        return JSONResponse(content={"summary": utils.get_results(file)})
    except Exception as e:
        print(f"Error during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


@app.post("/retrieve_and_generate")
async def retrieve_and_generate(image: UploadFile = File(...)):
    """Endpoint that: 1) retrieves top-k reports using MedSigLIP+FAISS and 2) runs MedGemma with those reports as context.

    index_path and meta_path should be accessible on the server filesystem (or mounted).
    """

    image_data = await image.read()
    try:
        result = llm_services.retrieve_then_generate(
            image_data=image_data,
            index_path=index_path,
            meta_path=meta_path,
            medsiglip_model="google/medsiglip-448",
            medgemma_variant="4b-it",
            k=5,
            hf_token=os.getenv("HF_TOKEN"),
        )
        try:
            print("Retrieve and Generate Result:", result)

            return JSONResponse(content={"result": result})
        except Exception as e:
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed retrieve+generate: {e}")


@app.post("/corrections")
async def process_corrections(request: CorrectionRequest):
    """
    Receives feedback, calculates a trust score, regenerates the summary,
    saves the final report to the database, and returns the new summary.
    """
    try:
        trust_score, regenerated_summary = (
            llm_services.calculate_trust_score_and_regenerate(
                original_summary=request.original_summary, feedback=request.feedback
            )
        )

        # Create and save the report to MongoDB
        new_report = Report(
            category=random.choice(["X-Ray", "MRI", "General", "Lab"]),
            input_type="Image" if "Image" in request.original_summary else "Text",
            trust_score=trust_score,
            summary=regenerated_summary,
            details=request.original_summary,
        )
        new_report.save()

        return JSONResponse(content={"summary": {"output": regenerated_summary}})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process corrections: {e}"
        )


@app.post("/accept")
async def process_acceptance(request: AcceptReportRequest):
    """
    saves the final report to the database with full trust considering the clinician downloaded the report.
    """
    try:
        # Create and save the report to MongoDB
        new_report = Report(
            category=random.choice(["X-Ray", "MRI", "General", "Lab"]),
            input_type="Image" if "Image" in request.summary else "Text",
            trust_score=100,
            summary=request.summary,
            details=request.summary,
        )
        new_report.save()

        return JSONResponse(
            content={"message": "Report accepted and saved successfully."}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process corrections: {e}"
        )


@app.post("/diagnose/download")
async def download_report(request: DownloadRequest):
    """
    Converts a given summary text into a downloadable PDF file.
    """
    try:
        pdf_bytes = utils.create_pdf_from_summary(request.summary)
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=diagnosis_summary.pdf"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create PDF: {e}")


@app.get("/patient_history")
async def get_metadata(patient_id: str, max_entries: Optional[int] = None):
    """
    Fetches a patient's precomputed context string (or object) loaded at startup.

    Query params:
    - patient_id: required patient identifier
    - max_entries: optional integer to truncate the history portion to the most-recent N entries
    """
    try:
        metadata_store = getattr(app.state, "METADATA_HISTORY", {})
        entry = metadata_store.get(patient_id)
        if entry is None:
            return JSONResponse(
                status_code=404,
                content={"error": "No data found for the given patient ID."},
            )

        # # If no truncation requested, return stored entry as-is
        # if max_entries is None:
        #     return JSONResponse(content={"metadata": entry})

        # # If the stored entry is a plain context string (format: "Meta Data:\n...\n\nHistory:\n<entries...>")
        # if isinstance(entry, str):
        #     parts = entry.split("\n\nHistory:\n", 1)
        #     meta_part = parts[0]
        #     history_part = parts[1] if len(parts) > 1 else ""
        #     entries = [e.strip() for e in history_part.split("\n\n") if e.strip()]
        #     truncated = "\n\n".join(entries[:max_entries])
        #     new_context = meta_part + "\n\nHistory:\n" + truncated
        #     return JSONResponse(content={"metadata": new_context})

        # # If the stored entry is structured (dict) with a 'history' list, truncate that list
        # if isinstance(entry, dict):
        #     hist = entry.get("history")
        #     if isinstance(hist, list):
        #         # Try to sort by parsed date descending if available, otherwise keep order
        #         def _get_date(h):
        #             return h.get("_parsed_date") or h.get("date") or ""

        #         try:
        #             sorted_hist = sorted(hist, key=lambda x: _get_date(x) or "", reverse=True)
        #         except Exception:
        #             sorted_hist = hist

        #         truncated_hist = sorted_hist[:max_entries]
        #         new_entry = dict(entry)
        #         new_entry["history"] = truncated_hist
        #         return JSONResponse(content={"metadata": new_entry})

        # Fallback: return the raw entry if we don't know how to truncate it
        return JSONResponse(content={"metadata": entry})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {e}")


@app.get("/reports", response_model=List[ReportResponse])
async def get_reports():
    """
    Fetches all saved reports from the database for the Reporting and
    Evaluations pages.
    """
    try:
        reports = Report.objects().order_by("-generated_date")
        return [report.to_dict() for report in reports]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reports: {e}")


# To run this application:
# 1. Create a .env file with your MONGO_URI and GOOGLE_API_KEY.
# 2. Run `uvicorn app:app --reload` in your terminal.
