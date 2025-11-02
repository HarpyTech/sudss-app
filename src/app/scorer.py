"""
tcs_genai_runner.py

Call Gemini Flash(-Lite) to compute Trust Calibration Score (TCS) with a strict system instruction (exact formulas).
Validate model output and fall back to deterministic local computation if needed.
"""

import os
import json
import math
import numpy as np
from typing import List, Optional, Dict

# Optional local fallbacks
try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import Levenshtein
    _HAS_LEV = True
except Exception:
    _HAS_LEV = False

# Try to import official Google GenAI client(s)
HAS_GENAI = False
genai = None
try:
    # preferred modern import pattern
    from google import genai as genai_pkg  # type: ignore
    genai = genai_pkg
    HAS_GENAI = True
except Exception:
    try:
        import google.generativeai as genai_pkg  # type: ignore
        genai = genai_pkg
        HAS_GENAI = True
    except Exception:
        HAS_GENAI = False

# --------- Deterministic local implementations (for fallback/validation) ----------
def _softmax(x: np.ndarray, temp: float = 0.1) -> np.ndarray:
    x = np.array(x, dtype=float)
    x_scaled = x / float(temp)
    x_shift = x_scaled - np.max(x_scaled)
    e = np.exp(x_shift)
    return e / e.sum()

def retrieval_confidence_from_scores(scores: List[float], temp: float = 0.1) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    inds = np.argsort(-arr)
    arr_sorted = arr[inds]
    if arr_sorted.shape[0] == 1:
        margin_conf = 1.0
    else:
        margin = float(arr_sorted[0] - arr_sorted[1])
        margin_conf = 1.0 / (1.0 + math.exp(-margin / (temp if temp>0 else 1e-6)))
    probs = _softmax(arr_sorted, temp=temp)
    H = -np.sum(probs * np.log(probs + 1e-12))
    maxH = math.log(len(probs)) if len(probs) > 1 else 1.0
    entropy_conf = 1.0 - (H / maxH)
    S_r = 0.7 * margin_conf + 0.3 * entropy_conf
    return float(max(0.0, min(1.0, S_r)))

def compute_semantic_similarity_local(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    if not _HAS_ST:
        # fallback: simple token overlap proxy (very rough)
        a = set(text1.lower().split())
        b = set(text2.lower().split())
        if not a or not b:
            return 0.0
        overlap = len(a & b) / max(len(a | b), 1)
        return float(max(0.0, min(1.0, overlap)))
    model = SentenceTransformer(model_name)
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    cos = util.cos_sim(emb1, emb2).item()
    cos01 = (cos + 1.0) / 2.0
    return float(max(0.0, min(1.0, cos01)))

def normalized_edit_distance(d1: str, d2: str) -> float:
    if _HAS_LEV:
        ed = Levenshtein.distance(d1, d2)
        denom = max(len(d1), len(d2), 1)
        return float(max(0.0, min(1.0, ed / denom)))
    else:
        # difflib ratio fallback
        import difflib
        ratio = difflib.SequenceMatcher(None, d1, d2).ratio()
        return float(max(0.0, min(1.0, 1.0 - ratio)))

def compute_tcs_local(
    retrieval_scores: List[float],
    draft_text: str,
    clinician_final: str,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    edit_method: str = "levenshtein"
) -> Dict:
    S_r = retrieval_confidence_from_scores(retrieval_scores)
    S_s = compute_semantic_similarity_local(draft_text, clinician_final, model_name=embedding_model_name)
    S_e = normalized_edit_distance(draft_text, clinician_final) if edit_method == "levenshtein" else normalized_edit_distance(draft_text, clinician_final)
    TCS = alpha * S_r + beta * S_s + gamma * (1.0 - S_e)
    TCS = float(max(0.0, min(1.0, TCS)))
    return {
        "S_r": round(S_r,6), "S_s": round(S_s,6), "S_e": round(S_e,6), "TCS": round(TCS,6),
        "method_S_e": edit_method, "notes":"local fallback"
    }

# ----------------- Strict system instruction (exact formulas & normalization/clamping) -----------------
SYSTEM_INSTRUCTION = r"""
You are a precise numeric calculator. Compute Trust Calibration Score (TCS) exactly as specified below.
Return EXACTLY one JSON object (no surrounding text). Numeric values only (not strings) for S_r, S_s, S_e, TCS.
All values must be in the closed interval [0.0, 1.0]. If any intermediate calculation risks division by zero, set that component to 0.0 and note it in "notes".

FORMULAS and NORMALIZATION:
  1) TCS = alpha * S_r + beta * S_s + gamma * (1 - S_e)
     - Defaults: alpha=0.4, beta=0.4, gamma=0.2 (unless explicitly overridden in the input).
     - After computing TCS, clamp to [0.0, 1.0].

  2) Retrieval confidence S_r in [0,1]:
     - Input: a list RETRIEVAL_SCORES = [s0, s1, ..., s_{K-1}] (floats; K>=1).
     - Use temp = 0.1.
     - margin_conf = sigmoid( (s0 - s1) / temp ) if K >= 2; if K == 1 set margin_conf = 1.0.
       where sigmoid(x) = 1 / (1 + exp(-x))
     - For entropy_conf:
         probs = softmax(RETRIEVAL_SCORES / temp)
         H = - sum_i probs[i] * ln(probs[i] + 1e-12)
         maxH = ln(K)  (if K>1) else 1.0
         entropy_conf = 1 - (H / maxH)
       If K==1 set entropy_conf = 1.0.
     - S_r = 0.7 * margin_conf + 0.3 * entropy_conf
     - Clamp S_r to [0.0, 1.0].

  3) Semantic similarity S_s in [0,1]:
     - S_s is semantic similarity between DRAFT and CLINICIAN_FINAL (1.0 identical meaning, 0.0 completely different).
     - You may compute S_s using embeddings & cosine similarity and then normalize from [-1,1] to [0,1] by: (cos + 1)/2.
     - If you cannot compute embeddings or numeric cosine, you must fallback to a token-overlap heuristic but state that in "notes".
     - Clamp S_s to [0.0, 1.0].

  4) Edit penalty S_e in [0,1]:
     - Preferred method: normalized Levenshtein distance = (Levenshtein distance between DRAFT and CLINICIAN_FINAL) / max(len(DRAFT), len(CLINICIAN_FINAL), 1).
     - Alternative (semantic): S_e = 1 - BERTScore_F1 (if available).
     - You must state which method you used in "method_S_e".
     - Clamp S_e to [0.0, 1.0]. 0.0 => identical, 1.0 => totally different.

DRAFT selection rules (apply before computing similarities):
  - If PREVIOUS_REPORT is provided and non-empty, DRAFT = PREVIOUS_REPORT.
  - Else DRAFT = GENERATED_REPORT.
  - If CLINICIAN_FINAL_REPORT is empty or missing, treat it as fully accepted and set CLINICIAN_FINAL_REPORT = GENERATED_REPORT.

INPUT (provided in the user content): You will receive:
  - RETRIEVAL_SCORES: JSON list of floats (top-K)
  - GENERATED_REPORT: text
  - CLINICIAN_FINAL_REPORT: text (may be empty)
  - PREVIOUS_REPORT: text (may be empty)
  - optional ALPHA, BETA, GAMMA (floats). If not present use default weights above.

OUTPUT JSON schema (exact keys and types):
{
  "S_r": number,
  "S_s": number,
  "S_e": number,
  "TCS": number,
  "draft_used": "prev_generated_report" OR "generated_report",
  "method_S_e": "levenshtein" OR "bertscore" OR "semantic_heuristic",
  "intermediates": {
     "margin_conf": number,
     "entropy_conf": number
  },
  "notes": string  // short note <= 120 chars
}

Important: Return ONLY the JSON object. No explanations, no extra text, no markdown.
Use temperature = 0.0 when generating.
"""

# ----------------- User prompt template (fill with actual data) -----------------
USER_PROMPT_TEMPLATE = """
RETRIEVAL_SCORES: {scores_json}

GENERATED_REPORT:
{generated}

CLINICIAN_FINAL_REPORT:
{clinician}

PREVIOUS_REPORT:
{previous}

# Optionally you may include ALPHA, BETA, GAMMA as JSON floats. If not included, use defaults.
"""

# ----------------- JSON validation / sanitizer -----------------
def validate_and_sanitize_model_json(obj: Dict) -> Dict:
    required = ("S_r","S_s","S_e","TCS","draft_used","method_S_e","intermediates","notes")
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing required key: {k}")
    # numeric checks
    for key in ("S_r","S_s","S_e","TCS"):
        val = obj[key]
        if not isinstance(val, (int,float)) or math.isnan(val):
            raise ValueError(f"{key} must be numeric")
        # clamp
        obj[key] = float(max(0.0, min(1.0, float(val))))
    # intermediates checks
    inter = obj["intermediates"]
    if not isinstance(inter, dict) or "margin_conf" not in inter or "entropy_conf" not in inter:
        raise ValueError("intermediates must contain margin_conf and entropy_conf")
    obj["draft_used"] = str(obj["draft_used"])
    if obj["draft_used"] not in ("prev_generated_report","generated_report"):
        raise ValueError("draft_used must be 'prev_generated_report' or 'generated_report'")
    if obj["method_S_e"] not in ("levenshtein","bertscore","semantic_heuristic"):
        # allow alternative labels but normalize
        obj["method_S_e"] = "semantic_heuristic"
    # notes length
    obj["notes"] = (str(obj["notes"])[:120]) if "notes" in obj else ""
    return obj

# ----------------- Main runner function -----------------
def run_tcs_via_genai(
    retrieval_scores: List[float],
    generated_report: str,
    clinician_final_report: Optional[str] = None,
    previous_report: Optional[str] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    model_name: str = "gemini-1.5-flash-lite",
    api_key_envvar: str = "GOOGLE_API_KEY"
) -> Dict:
    # Build user prompt
    scores_json = json.dumps(retrieval_scores)
    user_prompt = USER_PROMPT_TEMPLATE.format(scores_json=scores_json,
                                             generated=generated_report,
                                             clinician=clinician_final_report or "",
                                             previous=previous_report or "")
    # Add weights if provided
    if alpha is not None and beta is not None and gamma is not None:
        user_prompt += f"\nALPHA: {alpha}\nBETA: {beta}\nGAMMA: {gamma}\n"

    # call model if available
    if HAS_GENAI:
        key = os.environ.get(api_key_envvar)
        if not key:
            raise RuntimeError(f"No API key in env var {api_key_envvar}. Set your AI Studio API key.")
        # Configure genai client depending on package shape
        try:
            # new style: google.genai.configure
            if hasattr(genai, "configure"):
                genai.configure(api_key=key)
        except Exception:
            pass

        # Attempt to call generate with deterministic output
        try:
            # Many SDK versions: genai.generate(...) or genai.models.generate(...)
            if hasattr(genai, "generate"):
                resp = genai.generate(
                    model=model_name,
                    input=[
                        {"role":"system","content":SYSTEM_INSTRUCTION},
                        {"role":"user","content":user_prompt}
                    ],
                    temperature=0.0,
                    max_output_tokens=512
                )
                # Extract text field vary by SDK
                text = None
                if isinstance(resp, dict) and "candidates" in resp:
                    text = resp["candidates"][0]["content"]
                else:
                    # try attributes
                    text = getattr(resp, "text", None) or str(resp)
            elif hasattr(genai, "models") and hasattr(genai.models, "generate"):
                resp = genai.models.generate(
                    model=model_name,
                    messages=[{"role":"system","content":SYSTEM_INSTRUCTION},
                              {"role":"user","content":user_prompt}],
                    temperature=0.0,
                    max_output_tokens=512
                )
                # sdk shape: resp.candidates[0].content or resp.output[0].content
                text = None
                if hasattr(resp, "candidates"):
                    text = resp.candidates[0].content
                elif hasattr(resp, "output") and resp.output:
                    text = resp.output[0].content
                else:
                    text = str(resp)
            else:
                raise RuntimeError("Unrecognized genai client interface in this environment.")
        except Exception as e:
            # model call failure, fallback to deterministic
            print("GenAI call failed:", e)
            return fallback_local_compute(retrieval_scores, generated_report, clinician_final_report, previous_report, alpha,beta,gamma)

        # parse returned text as JSON
        try:
            parsed = json.loads(text.strip())
            sanitized = validate_and_sanitize_model_json(parsed)
            # final check: consistency of TCS with formula; if mismatch > 1e-3, note it but accept model value
            try:
                a = alpha if alpha is not None else 0.4
                b = beta if beta is not None else 0.4
                g = gamma if gamma is not None else 0.2
                recomputed = a * sanitized["S_r"] + b * sanitized["S_s"] + g * (1.0 - sanitized["S_e"])
                recomputed = float(max(0.0, min(1.0, recomputed)))
                if abs(recomputed - sanitized["TCS"]) > 1e-3:
                    sanitized["notes"] = (sanitized.get("notes","") + " | TCS mismatch with recomputed; recomputed in notes")[:120]
                    sanitized["recomputed_TCS"] = round(recomputed,6)
                return sanitized
            except Exception:
                return sanitized
        except Exception as e:
            print("Failed to parse/validate model JSON:", e)
            print("Model raw output:\n", text)
            # fallback to deterministic local
            return fallback_local_compute(retrieval_scores, generated_report, clinician_final_report, previous_report, alpha,beta,gamma)
    else:
        raise RuntimeError("No GenAI client available. Install 'google-genai' or 'google-generativeai' or run local fallback.")

def fallback_local_compute(retrieval_scores, generated_report, clinician_final_report, previous_report, alpha,beta,gamma):
    # Apply draft rules
    clinician = clinician_final_report if clinician_final_report and clinician_final_report.strip() else generated_report
    if previous_report and previous_report.strip():
        draft = previous_report
        draft_used = "prev_generated_report"
    else:
        draft = generated_report
        draft_used = "generated_report"
    a = alpha if alpha is not None else 0.4
    b = beta if beta is not None else 0.4
    g = gamma if gamma is not None else 0.2
    local = compute_tcs_local(retrieval_scores, draft, clinician, alpha=a, beta=b, gamma=g)
    local["draft_used"] = draft_used
    return local

# ----------------- Example usage -----------------
if __name__ == "__main__":
    # sample inputs
    retrieval_scores = [0.8822, 0.8562, 0.8516, 0.8481, 0.8420]
    generated = ("The image is a lateral chest X-ray. The lungs appear clear. The cardiac and mediastinal silhouettes are unremarkable. "
                 "There is no pneumothorax or pleural effusion.")
    clinician_corrected = ("Lateral chest radiograph: Lungs clear. Cardiomediastinal silhouette within normal size and contour. "
                          "No pneumothorax or pleural effusion identified. Impression: No acute cardiopulmonary disease.")
    previous = ""  # or provide prev draft text

    result = run_tcs_via_genai(retrieval_scores, generated, clinician_corrected, previous, model_name="gemini-1.5-flash-lite")
    print(json.dumps(result, indent=2))
