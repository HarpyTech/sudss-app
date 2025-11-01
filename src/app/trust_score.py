"""
trust_calibration_with_genai.py

Compute Trust Calibration Score (TCS) with optional Google Generative AI (GenAI) embedding support
(using AI Studio API key / google-genai SDK). Falls back to sentence-transformers if GenAI not used.

Reference/formula: See uploaded Trust calibration_score formula.pdf. :contentReference[oaicite:5]{index=5}
"""

from typing import List, Optional, Dict
import numpy as np
import math
import os
from dotenv import load_dotenv
# import levenshtein
from sentence_transformers import SentenceTransformer, util
from Levenshtein import distance as lev_distance
from bert_score import score as bertscore_score
import difflib
import google.generativeai as genai

load_dotenv()  # Load environment variables from .env file if present
# Local embedding (sentence-transformers)
try:
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Levenshtein
try:
    _HAS_LEV = True
except Exception:
    _HAS_LEV = False

# BERTScore (optional)
try:
    _HAS_BS = True
except Exception:
    _HAS_BS = False


def _softmax(x: np.ndarray, temp: float = 0.1) -> np.ndarray:
    x = np.array(x, dtype=float)
    x_scaled = x / float(temp)
    x_shift = x_scaled - np.max(x_scaled)
    e = np.exp(x_shift)
    return e / e.sum()


def retrieval_confidence_from_scores(scores: List[float], temp: float = 0.1) -> float:
    """Compute retrieval confidence S_r from a list of retrieval/hybrid scores."""
    if len(scores) == 0:
        return 0.0
    arr = np.array(scores, dtype=float)
    inds = np.argsort(-arr)
    arr_sorted = arr[inds]
    # margin confidence
    if arr_sorted.shape[0] == 1:
        margin_conf = 1.0
    else:
        margin = float(arr_sorted[0] - arr_sorted[1])
        margin_conf = 1.0 / (1.0 + math.exp(-margin / (temp if temp > 0 else 1e-6)))
    # entropy confidence
    probs = _softmax(arr_sorted, temp=temp)
    H = -np.sum(probs * np.log(probs + 1e-12))
    maxH = math.log(len(probs)) if len(probs) > 1 else 1.0
    entropy_conf = 1.0 - (H / maxH)
    # mixture weights (spec): 0.7 margin + 0.3 entropy
    S_r = 0.7 * margin_conf + 0.3 * entropy_conf
    return float(max(0.0, min(1.0, S_r)))


def compute_semantic_similarity_local(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Compute cosine similarity [0,1] using sentence-transformers (local)."""
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers not installed. Install via `pip install sentence-transformers`.")
    model = SentenceTransformer(model_name)
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    cos = util.cos_sim(emb1, emb2).item()  # [-1,1]
    cos01 = (cos + 1.0) / 2.0
    return float(max(0.0, min(1.0, cos01)))


def normalized_edit_distance(d1: str, d2: str) -> float:
    """Normalized Levenshtein distance: 0 identical -> 0, 1 totally different -> 1."""
    if _HAS_LEV:
        ed = lev_distance(d1, d2)
        denom = max(len(d1), len(d2), 1)
        return float(max(0.0, min(1.0, ed / denom)))
    else:

        ratio = difflib.SequenceMatcher(None, d1, d2).ratio()
        return float(max(0.0, min(1.0, 1.0 - ratio)))


def bertscore_penalty(draft: str, clinician: str, lang: str = "en") -> float:
    """Alternative edit penalty using BERTScore: returns value in [0,1]."""
    if not _HAS_BS:
        raise RuntimeError("bert-score not installed. Install via `pip install bert-score`.")
    P, R, F1 = bertscore_score([draft], [clinician], lang=lang, verbose=False)
    f1 = float(F1[0].item())
    return float(max(0.0, min(1.0, 1.0 - f1)))


# --------------------------
# Google GenAI embedding helper (AI Studio / google-genai):
# --------------------------
def compute_semantic_similarity_google_genai(text1: str, text2: str, api_key: Optional[str] = None,
                                             model: str = "text-embedding-004") -> float:
    """
    Compute embedding cosine similarity using Google GenAI SDK / API.
    This function attempts a few common call patterns. If none are available, it raises a clear error.

    Notes:
    - Set your AI Studio API key in environment variable: GOOGLE_API_KEY (or pass api_key param).
    - Official SDK packages: 'google-genai' (preferred) or older 'google-generativeai'.
    - Example SDK docs: google-genai / python-genai. See references in the function docstrings.
    - The exact model name (text-embedding-3-large, gemini-embedding-001, text-embedding-004, etc.) depends on your account & available models.
    """

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No Google AI Studio API key found. Set the GOOGLE_API_KEY env var or pass api_key.")

    try:

        genai.configure(api_key=key)
        # Use the generate_embeddings method if available
        emb1_resp = genai.embed_content(model=model, content=text1)
        emb2_resp = genai.embed_content(model=model, content=text2)
        v1 = np.array(emb1_resp['embedding'])
        v2 = np.array(emb2_resp['embedding'])
            
        # cosine similarity
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        cos01 = (cos + 1.0) / 2.0
        return float(max(0.0, min(1.0, cos01)))

    except Exception as exc:
        raise RuntimeError(
            "Could not call Google GenAI embedding client from this environment. "
            "Install and configure the official SDK (pip install google-genai) and set GOOGLE_API_KEY. "
            "See: https://googleapis.github.io/python-genai/ and https://ai.google.dev/gemini-api/docs/embeddings for examples. "
            f"Underlying error: {exc}"
        ) from exc


# --------------------------
# Main TCS function (follows your acceptance/draft selection rules)
# --------------------------
def compute_tcs(
    retrieval_scores: List[float],
    generated_report: str,
    clinician_final_report: Optional[str] = None,
    prev_generated_report: Optional[str] = None,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    use_google_genai: bool = False,
    google_embedding_model: str = "text-embedding-004",
    embedding_model_name_local: str = "all-MiniLM-L6-v2",
    edit_penalty_method: str = "levenshtein"
) -> Dict:
    """
    Compute the Trust Calibration Score with the rules:
    - If prev_generated_report provided -> use it as draft_text
    - Else use generated_report
    - If clinician_final_report is None/empty -> treat as fully accepted (clinician_final_report = generated_report)
    Returns a dict with S_r, S_s, S_e, TCS, plus metadata
    """
    # weight check
    if abs((alpha + beta + gamma) - 1.0) > 1e-6:
        raise ValueError("alpha + beta + gamma must equal 1.0")

    # Acceptance rule
    if clinician_final_report is None or str(clinician_final_report).strip() == "":
        clinician_final_report = generated_report

    # Choose draft according to your rule
    if prev_generated_report is not None and str(prev_generated_report).strip() != "":
        draft_text = prev_generated_report
        draft_used = "prev_generated_report"
    else:
        draft_text = generated_report
        draft_used = "generated_report"

    # 1) retrieval confidence
    S_r = retrieval_confidence_from_scores(retrieval_scores)

    # 2) semantic similarity S_s
    if use_google_genai:
        S_s = compute_semantic_similarity_google_genai(draft_text, clinician_final_report,
                                                      model=google_embedding_model)
    else:
        S_s = compute_semantic_similarity_local(draft_text, clinician_final_report, model_name=embedding_model_name_local)

    # 3) edit penalty S_e
    if edit_penalty_method == "levenshtein":
        S_e = normalized_edit_distance(draft_text, clinician_final_report)
    elif edit_penalty_method == "bertscore":
        S_e = bertscore_penalty(draft_text, clinician_final_report)
    else:
        raise ValueError("Unknown edit_penalty_method")

    # Final TCS
    TCS = alpha * S_r + beta * S_s + gamma * (1.0 - S_e)
    TCS = float(max(0.0, min(1.0, TCS)))

    return {
        "S_r": round(float(S_r), 6),
        "S_s": round(float(S_s), 6),
        "S_e": round(float(S_e), 6),
        "TCS": round(float(TCS), 6),
        "draft_used": draft_used,
        "clinician_accepted_generated": bool(clinician_final_report == generated_report)
    }


import re

def test():
    text = """
    You are a radiology assistant. Use the example reports below as reference.

    --- Example 1 | uid: 22 | projection: Lateral | score: 0.8822 ---
    Findings: The lungs are clear, and without focal air space opacity. The cardiomediastinal silhouette is normal in size and contour, and stable. There is no pneumothorax large pleural effusion.
    Impression: No acute cardiopulmonary abnormality.

    --- Example 2 | uid: 20 | projection: Lateral | score: 0.8562 ---
    Findings: The cardiac and mediastinal silhouettes are unremarkable. The lungs are well expanded and clear. There are no focal air space opacities. There is no pneumothorax or effusion. There are mild degenerative changes of the thoracic spine.
    Impression: No evidence of acute cardiopulmonary process. Stable appearance of the chest.

    --- Example 3 | uid: 17 | projection: Lateral | score: 0.8516 ---
    Findings: No focal areas of consolidation. No suspicious pulmonary opacities. Heart size within normal limits. No pleural effusions. No evidence of pneumothorax. Osseous structures intact.
    Impression: No acute cardiopulmonary abnormality.

    --- Example 4 | uid: 3 | projection: Lateral | score: 0.8481 ---
    Findings: nan
    Impression: No displaced rib fractures, pneumothorax, or pleural effusion identified. Well-expanded and clear lungs. Mediastinal contour within normal limits. No acute cardiopulmonary abnormality identified.

    --- Example 5 | uid: 12 | projection: Lateral | score: 0.8420 ---
    Findings: Lungs are clear bilaterally. Cardiac and mediastinal silhouettes are normal. Pulmonary vasculature is normal. No pneumothorax or pleural effusion. No acute bony abnormality.
    Impression: No acute cardiopulmonary abnormality.


            Furthermore, please consider the following additional information as clinician notes:
    """

    # Use regex to find all occurrences of "score: " followed by a number
    scores = re.findall(r"score: (\d+\.\d+)", text)

    # Convert the extracted strings to floats
    retrieval_scores = [float(score) for score in scores]

    print("Extracted Retrieval Scores:", retrieval_scores)

    # Assuming 'compute_tcs' function is defined in the previous cell
    # You can now use these extracted scores with the compute_tcs function
    # For example:
    generated = "The image is a lateral chest X-ray. The lungs appear clear. The cardiac and mediastinal silhouettes are unremarkable. There is no pneumothorax or pleural effusion.\n\n### **Patient Name:** Test Patient\n## Summary on the X-ray Image:\nThe lateral chest X-ray shows clear lungs, normal cardiac and mediastinal silhouettes, and no evidence of pneumothorax or pleural effusion.\n## Overall Summary:\nNo previous history available to compare.\n### Findings\n*   Lungs are clear bilaterally.\n*   Cardiac and mediastinal silhouettes are normal.\n*   No pneumothorax or pleural effusion.\n### Impression\n*   No acute cardiopulmonary abnormality.\n"
    # Your generated report text
    clinician_corrected = "All Good and report is good to go" # Your clinician's corrected report text (optional)
    prev_generated = generated  # Your previous generated report text (optional)

    out = compute_tcs(retrieval_scores, generated_report=generated,
                    clinician_final_report=clinician_corrected,
                    prev_generated_report=prev_generated,
                    use_google_genai=True)
    print(out)


if __name__ == "__main__":
    test()