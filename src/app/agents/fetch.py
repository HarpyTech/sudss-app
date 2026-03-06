# app.py
"""
FastAPI wrapper for retrieve_and_summarize.py
- All configuration is hardcoded (no dynamic CLI or env vars).
- Endpoint: POST /summarize (multipart/form-data) with field "file"
"""

from typing import List, Dict, Any
from pathlib import Path
from io import BytesIO
import json

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import faiss
import os
import sys
import logging

# Use the shared logger so every log line carries file/function/line metadata
# that the RCA Agent can use for code lookup.
try:
    from logger_config import get_logger
except ImportError:
    # Fallback when the module is imported standalone (e.g. in unit tests)
    logging.basicConfig(
        format=(
            "%(asctime)s - %(name)s - %(levelname)s"
            " - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"
        ),
        stream=sys.stdout,
        level=logging.INFO,
    )
    get_logger = logging.getLogger

logger = get_logger(__name__)

# --------------------------
# Hardcoded configuration
# --------------------------
INDEX_PATH = os.path.abspath("./src/embeddings_25k/images.index")  # hardcoded index file
META_PATH = os.path.abspath("./src/embeddings_25k/metadata.jsonl")  # hardcoded metadata jsonl
MEDSIGLIP_MODEL = "google/medsiglip-448"  # medsiglip model used for embeddings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5
MAX_CONTEXT_REPORTS = 5
MAX_NEW_TOKENS = 256
PROMPT_TRUNCATE_PREVIEW = 1000

# --------------------------
# App & global state
# --------------------------
FAISS_INDEX = None
METADATA: List[Dict[str, Any]] = []
EMBEDDER = None
GEN_TOKENIZER = None
GEN_MODEL = None


# --------------------------
# Utilities
# --------------------------
def load_metadata_jsonl(path: str) -> List[Dict[str, Any]]:
    meta = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"metadata file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


class MedSigLIPEmbedder:
    """Small helper to produce an image embedding for a query image using MedSigLIP-like model."""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()

    def embed_image_pil(self, pil_image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            try:
                feats = self.model.get_image_features(**inputs)
            except Exception:
                out = self.model(**inputs)
                if hasattr(out, "image_embeds"):
                    feats = out.image_embeds
                elif hasattr(out, "pooler_output"):
                    feats = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError("Could not extract image features from model output")
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(np.asarray(feats)).to(self.device)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        emb = feats.cpu().numpy().astype("float32")
        return emb

def retrive_topk_hybrid(q_emb: np.ndarray, k: int):
    """
    Hybrid retrieval using multiple similarity measures with thresholding.
    Combines cosine similarity, dot product similarity, and inverse Euclidean distance.
    Returns top-k results above a defined threshold.
    """
    global FAISS_INDEX
        # ==========================================================
    # 2️⃣ STRONG HYBRID RETRIEVAL WITH THRESHOLD (IMAGE-ONLY)
    # ==========================================================
    # db_embs = FAISS_INDEX.reconstruct_n(0, FAISS_INDEX.ntotal) # get all db embeddings
    db_embs = []
    for i in range(FAISS_INDEX.ntotal):
        emb = FAISS_INDEX.reconstruct(i)
        db_embs.append(emb)
    db_embs = np.array(db_embs, dtype=np.float32)

    logger.debug(
        "Database embeddings shape: %s, Query embedding shape: %s",
        db_embs.shape,
        q_emb.shape,
    )


    # Normalize
    db_embs = db_embs / np.linalg.norm(db_embs, axis=1, keepdims=True)
    q_emb_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    # Multiple similarity measures
    cos_sim = np.dot(db_embs, q_emb_norm.squeeze()) # Cosine similarity
    dot_sim = np.dot(db_embs, q_emb.squeeze()) # Dot product similarity
    l2_dist = np.linalg.norm(db_embs - q_emb, axis=1)
    inv_euc_sim = 1 / (1 + l2_dist) # Inverse Euclidean distance similarity

    # Weighted hybrid fusion
    alpha, beta, gamma = 0.5, 0.3, 0.2
    hybrid_score = alpha * cos_sim + beta * dot_sim + gamma * inv_euc_sim

    # Apply threshold
    logger.debug(
        "Hybrid scores range: min=%.4f max=%.4f",
        hybrid_score.min(),
        hybrid_score.max(),
    )
    threshold = 0.8
    # find indices where the hybrid score meets the threshold
    # np.all(...) returns a scalar for 1-D arrays, so use np.where to get indices
    valid_indices = np.where(hybrid_score >= threshold)[0]
    # fallback: if nothing meets threshold, consider all indices (so retrieval still returns something)
    if valid_indices.size == 0:
        valid_indices = np.arange(hybrid_score.shape[0])

    # Sort and select top-K
    logger.debug(
        "Total valid indices above threshold=%.2f: %d",
        threshold,
        len(valid_indices),
    )
    sorted_indices = valid_indices[np.argsort(hybrid_score[valid_indices])[::-1]]
    topk_indices = sorted_indices[:TOP_K]
    scores = hybrid_score[topk_indices]
    ids = topk_indices.tolist()

    logger.info(
        "retrive_topk_hybrid: retrieved %d results above threshold %.2f",
        len(topk_indices),
        threshold,
    )
    return scores, ids


def retrieve_topk(index, q_emb: np.ndarray, k: int):
    # index is a faiss index object (already loaded)
    k_search = min(k, index.ntotal)
    if k_search == 0:
        return [], []
    scores, ids = index.search(q_emb, k_search)
    return scores[0].tolist(), ids[0].tolist()


def assemble_prompt_from_reports(reports: List[Dict[str, Any]], query_projection: str = None) -> str:
    ctx_lines = []
    ctx_lines.append("You are a radiology assistant. Use the example reports below as reference.")
    ctx_lines.append("")
    for i, r in enumerate(reports, start=1):
        header = f"--- Example {i} | uid: {r.get('uid')} | projection: {r.get('projection')} | score: {r.get('_score'):.4f} ---"
        ctx_lines.append(header)
        f = str(r.get("findings") or "")
        im = str(r.get("impression") or "")
        max_len = 800
        # print(f"Finidings and Impressions before truncation: {f}  {im}", f, im)
        try:
            if len(f) > max_len:
                f = f[:max_len] + " ... [truncated]"
            if len(im) > max_len:
                im = im[:max_len] + " ... [truncated]"
        except Exception as e:
            logger.error(
                "Error truncating report text for uid=%s: %s",
                r.get("uid"),
                e,
                exc_info=True,
            )
        ctx_lines.append("Findings: " + f)
        ctx_lines.append("Impression: " + im)
        ctx_lines.append("")
    # ctx_lines.append("Now given the query chest x-ray image (use the same style as the examples),")
    # if query_projection:
    #     ctx_lines.append(f"Projection: {query_projection}")
    # ctx_lines.append("Write two sections clearly labeled 'Findings:' and 'Impression:' — be concise and mention uncertainty where appropriate.")
    ctx_lines.append("")

    logger.debug("assemble_prompt_from_reports: prompt assembled with %d reports.", len(reports))
    return "\n".join(ctx_lines)

# --------------------------
# Pydantic response models
# --------------------------
class RetrievedItem(BaseModel):
    uid: Any = None
    projection: Any = None
    findings: str = ""
    impression: str = ""
    _score: float = 0.0
    _rank: int = 0


class SummarizeResponse(BaseModel):
    query_image_name: str
    k_returned: int
    retrieved: List[RetrievedItem]
    prompt_preview: str
    generated: str


# --------------------------
# Startup
# --------------------------
def startup_load():
    global FAISS_INDEX, METADATA, EMBEDDER, GEN_TOKENIZER, GEN_MODEL, DEVICE

    meta_p = Path(META_PATH)
    if not meta_p.exists():
        logger.error("startup_load: metadata file not found at '%s'.", META_PATH)
        raise RuntimeError(f"Metadata file not found at {META_PATH}")
    METADATA = load_metadata_jsonl(META_PATH)
    logger.info("startup_load: loaded %d metadata records from '%s'.", len(METADATA), META_PATH)

    idx_p = Path(INDEX_PATH)
    if not idx_p.exists():
        logger.error("startup_load: FAISS index file not found at '%s'.", INDEX_PATH)
        raise RuntimeError(f"FAISS index file not found at {INDEX_PATH}")
    FAISS_INDEX = faiss.read_index(INDEX_PATH)
    logger.info(
        "startup_load: FAISS index loaded from '%s' (ntotal=%d).",
        INDEX_PATH,
        FAISS_INDEX.ntotal,
    )

    EMBEDDER = MedSigLIPEmbedder(MEDSIGLIP_MODEL, DEVICE)
    logger.info("startup_load: MedSigLIP embedder initialised (model=%s).", MEDSIGLIP_MODEL)


# --------------------------
# Endpoint: /summarize
# --------------------------
def summarize(file: bytes, patient_id: str, patient_context: str, 
              is_base_retrival: bool = False, corrections: str = None,
              previous_summary: str = None, text: str = None) -> SummarizeResponse:
    """
    Accept an image file and return retrieved examples + generated Findings/Impression.
    All behavior (k, context size, models, etc.) is hardcoded.
    """
    global FAISS_INDEX, METADATA, EMBEDDER

    if FAISS_INDEX is None or EMBEDDER is None:
        logger.error(
            "summarize: server not ready – FAISS_INDEX=%s EMBEDDER=%s",
            FAISS_INDEX,
            EMBEDDER,
        )
        raise HTTPException(status_code=503, detail="Server not ready (index or models not loaded).")

    # read uploaded image into PIL
    try:
        pil_img = Image.open(BytesIO(file)).convert("RGB")
    except Exception as e:
        logger.error("summarize: failed to decode uploaded image: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # 1) Embed
    q_emb = EMBEDDER.embed_image_pil(pil_img)


    # 2) Retrieve top-k
    # scores, ids = retrive_topk_hybrid(q_emb, TOP_K) if is_base_retrival else retrieve_topk(FAISS_INDEX, q_emb, TOP_K)
    if is_base_retrival == False:
        logger.info("summarize: using standard FAISS retrieval (patient_id=%s).", patient_id)
        scores, ids = retrieve_topk(FAISS_INDEX, q_emb, TOP_K)
    else:
        logger.info("summarize: using hybrid retrieval (patient_id=%s).", patient_id)
        scores, ids = retrive_topk_hybrid(q_emb, TOP_K)
    # 3) Map ids -> metadata
    results = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0 or idx >= len(METADATA):
            continue
        m = METADATA[idx].copy()
        m["_score"] = float(score)
        m.setdefault("findings", "")
        m.setdefault("impression", "")
        results.append(m)

    logger.info("summarize: retrieved %d reports from metadata.", len(results))

    # 4) Compose prompt
    top_n = min(MAX_CONTEXT_REPORTS, len(results))
    context_reports = results[:top_n]

    logger.debug("summarize: assembling prompt from %d context reports.", top_n)

    prompt = assemble_prompt_from_reports(context_reports, query_projection=None)

    # if is_base_retrival:
    prompt += f"""Now given the patient history context below
    — be concise and mention uncertainty where appropriate.
    Patient History Context:
    Patinet ID: {patient_id}
    {patient_context} """

    # if corrections:
    prompt += f"""
    Additionally, please incorporate the following corrections into your report:
    {corrections}

    Previous Summary:
    {previous_summary}
    """
    # if text:
    prompt += f"""
    Furthermore, please consider the following additional information as clinician notes:
    {text}
    """

    logger.info("summarize: prompt assembled successfully for patient_id=%s.", patient_id)

    return prompt
