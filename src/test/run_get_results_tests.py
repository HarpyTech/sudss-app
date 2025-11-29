import os
import time
import json
from datetime import datetime
import pathlib
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# run_get_results_tests.py
# - Reads `test_suite.csv` and `newfiles.csv` (both expected next to this script)
# - Creates a cross-join so each test-suite row is paired with each filename
# - POSTs `file_path` to POST /get_results
# - Records response status and body, writes periodic Excel checkpoints
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent
TEST_SUITE_CSV = BASE_DIR / "test_suite.csv"
NEWFILES_CSV = BASE_DIR / "newfiles.csv" if (BASE_DIR.parent / "newfiles.csv").exists() else BASE_DIR / "newfiles.csv"

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
GET_RESULTS_ENDPOINT = os.getenv("API_GET_RESULTS_ENDPOINT", "/get_results")

# Resilience params
DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))
RETRIES = int(os.getenv("API_RETRIES", "1"))
BACKOFF = float(os.getenv("API_BACKOFF", "2.0"))
CHECKPOINT_EVERY = int(os.getenv("API_CHECKPOINT_EVERY", "5"))


def load_inputs():
    if not TEST_SUITE_CSV.exists():
        raise FileNotFoundError(f"Test suite CSV not found: {TEST_SUITE_CSV}")
    if not NEWFILES_CSV.exists():
        raise FileNotFoundError(f"New files CSV not found: {NEWFILES_CSV}")

    df_tests = pd.read_csv(TEST_SUITE_CSV)
    df_files = pd.read_csv(NEWFILES_CSV)

    # Ensure the filename column exists
    if "filename" not in df_files.columns:
        raise ValueError("newfiles.csv must contain a 'filename' column")

    # Cross-join: pair every test row with every filename
    df_tests["output_file"] = df_files.filename
    # df_files["__key"] = 1
    df = df_tests # pd.merge(df_tests, df_files, on="__key").drop(columns=["__key"])

    # Prepare output columns
    df["response_status"] = ""
    df["response_body"] = ""
    df["timestamp"] = ""

    return df


def call_get_results(file_path: str):
    url = f"{API_BASE_URL.rstrip('/')}/{GET_RESULTS_ENDPOINT.lstrip('/')}"
    attempt = 0
    while attempt <= RETRIES:
        try:
            resp = requests.post(url, data={"file_path": file_path}, timeout=DEFAULT_TIMEOUT)
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:10000]
            return resp.status_code, body
        except Exception as e:
            attempt += 1
            if attempt > RETRIES:
                return "REQUEST_FAILED", str(e)
            sleep_for = BACKOFF * (2 ** (attempt - 1))
            print(f"Transient error calling get_results for {file_path}: {e}. Sleeping {sleep_for}s before retry.")
            time.sleep(sleep_for)


def main():
    df = load_inputs()
    total = len(df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = BASE_DIR / f"get_results_api_test_{timestamp}.xlsx"

    print(f"Starting tests: {total} requests to {API_BASE_URL}{GET_RESULTS_ENDPOINT}")

    completed = 0
    for idx, row in df.iterrows():
        filename = row.get("output_file", "")
        # Normalise filename: use as-is; server's utils.get_results likely expects repo-relative path
        print(f"[{idx + 1}/{total}] Calling get_results for file: {filename}")
        status, body = call_get_results(filename)

        df.at[idx, "response_status"] = status
        if isinstance(body, (dict, list)):
            df.at[idx, "response_body"] = json.dumps(body, ensure_ascii=False)
        else:
            df.at[idx, "response_body"] = str(body)
        df.at[idx, "timestamp"] = datetime.now().isoformat()

        completed += 1

        if completed % CHECKPOINT_EVERY == 0 or completed == total:
            try:
                df.to_excel(out_file, index=False)
                print(f"Checkpoint saved: {out_file} ({completed}/{total})")
            except Exception as e:
                print(f"Warning: failed to write checkpoint: {e}")

    # Final save
    df.to_excel(out_file, index=False)
    print(f"All tests done. Results saved to {out_file}")


if __name__ == "__main__":
    main()
