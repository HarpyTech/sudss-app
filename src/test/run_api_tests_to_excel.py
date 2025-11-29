import pandas as pd
import requests
import pathlib
import os
import json
import time
from datetime import datetime

# --- CONFIG ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/")
ENDPOINT = os.getenv("API_ENDPOINT", "diagnose")

# Use pathlib and default to a test_suite.csv located next to this script to avoid
# working-directory dependent paths.
BASE_DIR = pathlib.Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "test_suite.csv"
OUTPUT_FILE = BASE_DIR / f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

# --- Resilience settings ---
# Default total request timeout (seconds). Set via env API_TIMEOUT if needed.
DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "1800"))  # 30 minutes default
# Number of retries for transient failures
RETRIES = int(os.getenv("API_RETRIES", "0"))
# Backoff base seconds
BACKOFF = float(os.getenv("API_BACKOFF", "2.0"))
# How many tests between checkpoints
CHECKPOINT_EVERY = int(os.getenv("API_CHECKPOINT_EVERY", "5"))


def _safe_write(df, index, status, body):
    """Write results back to the dataframe. Keep it simple for sequential runs."""
    df.at[index, "response_status"] = status
    # store JSON as string to avoid Excel serialization issues
    if isinstance(body, (dict, list)):
        df.at[index, "response_body"] = json.dumps(body, ensure_ascii=False)
    else:
        df.at[index, "response_body"] = str(body)


def perform_test(index, row):
    """Runs a single test case with retries and returns (index, status, body)."""
    form_data = {
        "text": row.get("text", "Describe the given Image"),
        "is_base_retrival": str(row.get("is_base_retrival", "true")).lower(),
        "patient_id": row.get("patient_id", "P1001####"),
    }

    image_path = row.get("image_path") or ""
    img_path = None
    if isinstance(image_path, str) and image_path.strip():
        p = pathlib.Path(image_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists() and p.is_file():
            img_path = p

    attempt = 0
    while attempt <= RETRIES:
        try:
            print(f"Test index={index} attempt={attempt + 1}")
            if img_path:
                with open(str(img_path), "rb") as f:
                    files = {"image": f}
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
            print(f"Test index={index} completed with status={status}")
            return index, status, body

        except Exception as e:
            attempt += 1
            if attempt > RETRIES:
                print(f"Test index={index} failed after {RETRIES} retries: {e}")
                return index, "REQUEST_FAILED", str(e)
            # exponential backoff before retry
            sleep_for = BACKOFF * (2 ** (attempt - 1))
            print(f"Transient error, sleeping {sleep_for}s before retry: {e}")
            time.sleep(sleep_for)

    # unreachable
    return index, "REQUEST_FAILED", "Unknown error"


def main():
    # --- LOAD CSV ---
    df = pd.read_csv(INPUT_FILE)
    # df = df.head(1)  # For quick testing, remove or adjust as needed
    print(f"Loaded {len(df)} test cases from {INPUT_FILE}")

    # --- Prepare output columns ---
    df["response_status"] = ""
    df["response_body"] = ""

    total = len(df)
    completed = 0
    print(f"🚀 Starting API tests (sequential) timeout={DEFAULT_TIMEOUT}s, retries={RETRIES}")
    print(f"Timestamp: {datetime.now().isoformat()} at the start of tests.")

    for idx, row in df.iterrows():
        print(f"Processing test case {idx + 1}/{total} - Patient ID: {row.get('patient_id', 'N/A')}")
        index, status, body = perform_test(idx, row)
        _safe_write(df, index, status, body)
        completed += 1

        # checkpoint to disk every CHECKPOINT_EVERY completed tests
        if completed % CHECKPOINT_EVERY == 0 or completed == total:
            try:
                OUTPUT_FILE = BASE_DIR / f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                df.to_excel(OUTPUT_FILE, index=False)
                print(f"Checkpoint saved after {completed} tests to {OUTPUT_FILE}")
            except Exception as e:
                print(f"Warning: failed to write checkpoint: {e}")

    print(f"\n🚀 Timestamp: {datetime.now().isoformat()} at the end of tests.")
    print("All tests completed.")

    # --- SAVE RESULTS ---
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ All test cases executed. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
