import pandas as pd
import requests
import pathlib
import os
import json
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
API_BASE_URL = "http://localhost:8000/"
ENDPOINT = "diagnose"

# Use pathlib and default to a test_suite.csv located next to this script to avoid
# working-directory dependent paths.
BASE_DIR = pathlib.Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "test_suite.csv"
OUTPUT_FILE = BASE_DIR / f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

# --- Resilience / concurrency settings ---
# Default total request timeout (seconds). Set via env API_TIMEOUT if needed.
DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "1200"))  # 20 minutes default
# Number of parallel workers to run concurrently. Tunable via API_WORKERS env var.
WORKERS = int(os.getenv("API_WORKERS", "2"))
# Number of retries for transient failures
RETRIES = int(os.getenv("API_RETRIES", "0"))
# Backoff base seconds
BACKOFF = float(os.getenv("API_BACKOFF", "2.0"))

# --- LOAD CSV ---
df = pd.read_csv(INPUT_FILE)

# --- Prepare output columns ---
df["response_status"] = ""
df["response_body"] = ""

# Thread-safe write helpers
df_lock = threading.Lock()

def _safe_write(index, status, body):
    with df_lock:
        df.at[index, "response_status"] = status
        # store JSON as string to avoid Excel serialization issues
        if isinstance(body, (dict, list)):
            df.at[index, "response_body"] = json.dumps(body, ensure_ascii=False)
        else:
            df.at[index, "response_body"] = str(body)


def perform_test(index, row):
    """Runs a single test case with retries and returns (index, status, body)."""
    # Prepare multipart form data as per JS code
    form_data = {
        "text": row.get("text", "Describe the given Image"),
        "is_base_retrival": str(row.get("is_base_retrival", "true")).lower(),
        "patient_id": row.get("patient_id", "P1001####"),
    }

    # Resolve image path relative to BASE_DIR if not absolute
    files = None
    image_path = row.get("image_path") or ""
    if isinstance(image_path, str) and image_path.strip():
        img_path = pathlib.Path(image_path)
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path
        if img_path.exists() and img_path.is_file():
            files = {"image": open(str(img_path), "rb")}

        else:
            files = None

    attempt = 0
    while attempt <= RETRIES:
        try:
            resp = requests.post(
                f"{API_BASE_URL.rstrip('/')}/{ENDPOINT.lstrip('/')}",
                data=form_data,
                files=files,
                timeout=DEFAULT_TIMEOUT,
            )
            # Try parse JSON, else capture text (truncate)
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:5000]

            status = resp.status_code
            return index, status, body

        except Exception as e:
            attempt += 1
            if attempt > RETRIES:
                return index, "REQUEST_FAILED", str(e)
            # exponential backoff before retry
            sleep_for = BACKOFF * (2 ** (attempt - 1))
            time.sleep(sleep_for)
    # unreachable
    return index, "REQUEST_FAILED", "Unknown error"


# Run tests with a ThreadPoolExecutor to allow concurrent long-running requests.
with ThreadPoolExecutor(max_workers=WORKERS) as exec:
    futures = {exec.submit(perform_test, idx, row): idx for idx, row in df.iterrows()}
    # futures = {exec.submit(perform_test, idx, row): idx for idx, row in df.head(1).iterrows()}
    completed = 0
    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            index, status, body = fut.result()
        except Exception as e:
            index = idx
            status = "REQUEST_FAILED"
            body = str(e)

        _safe_write(index, status, body)
        completed += 1
        # checkpoint to disk every 10 completed tests
        if completed % 10 == 0 or completed == len(df):
            try:
                df.to_excel(OUTPUT_FILE, index=False)
                print(f"Checkpoint saved after {completed} tests to {OUTPUT_FILE}")
            except Exception as e:
                print(f"Warning: failed to write checkpoint: {e}")

# --- SAVE RESULTS ---
df.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ All test cases executed. Results saved to {OUTPUT_FILE}")
