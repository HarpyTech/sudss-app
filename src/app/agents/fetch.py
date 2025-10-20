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

# --------------------------
# Hardcoded configuration
# --------------------------
INDEX_PATH = os.path.abspath("./src/embeddings/images.index")          # hardcoded index file
META_PATH = os.path.abspath("./src/embeddings/metadata.jsonl")          # hardcoded metadata jsonl
MEDSIGLIP_MODEL = "google/medsiglip-448"              # medsiglip model used for embeddings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5                                             # top-k retrieval
MAX_CONTEXT_REPORTS = 5                               
MAX_NEW_TOKENS = 256                                  # generation length limit
PROMPT_TRUNCATE_PREVIEW = 1000                        # how many chars to return in prompt preview

# --------------------------
# App & global state
# --------------------------
# app = FastAPI(title="RetrieveAndSummarize (FAISS + MedSigLIP + text-gen)")

# globals to load at startup
FAISS_INDEX = None
METADATA: List[Dict[str, Any]] = []
EMBEDDER = None
GEN_TOKENIZER = None
GEN_MODEL = None


# --------------------------
# Utilities (converted from your script)
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
        # load the model that produces image features; trust_remote_code may be required
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
            emb = feats.cpu().numpy().astype("float32")  # shape (1, D)
        return emb


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
        f = r.get("findings") or ""
        im = r.get("impression") or ""
        max_len = 800
        if len(f) > max_len:
            f = f[:max_len] + " ... [truncated]"
        if len(im) > max_len:
            im = im[:max_len] + " ... [truncated]"
        ctx_lines.append("Findings: " + f)
        ctx_lines.append("Impression: " + im)
        ctx_lines.append("")
    # ctx_lines.append("Now given the query chest x-ray image (use the same style as the examples),")
    # if query_projection:
    #     ctx_lines.append(f"Projection: {query_projection}")
    # ctx_lines.append("Write two sections clearly labeled 'Findings:' and 'Impression:' — be concise and mention uncertainty where appropriate.")
    ctx_lines.append("")
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
# Startup: load index, metadata, embedder, generator
# --------------------------
def startup_load():
    global FAISS_INDEX, METADATA, EMBEDDER, GEN_TOKENIZER, GEN_MODEL, DEVICE

    # 1) Load metadata
    meta_p = Path(META_PATH)
    if not meta_p.exists():
        raise RuntimeError(f"Metadata file not found at {META_PATH}")
    METADATA = load_metadata_jsonl(META_PATH)

    # 2) Load FAISS index
    idx_p = Path(INDEX_PATH)
    if not idx_p.exists():
        raise RuntimeError(f"FAISS index file not found at {INDEX_PATH}")
    FAISS_INDEX = faiss.read_index(INDEX_PATH)

    # 3) Load MedSigLIP embedder
    EMBEDDER = MedSigLIPEmbedder(MEDSIGLIP_MODEL, DEVICE)

    # # 4) Load generation model & tokenizer (text-only)
    # GEN_TOKENIZER = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
    # GEN_MODEL = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME, trust_remote_code=True).to(DEVICE)
    # GEN_MODEL.eval()


# --------------------------
# Endpoint: /summarize
# --------------------------
def summarize(file: bytes) -> SummarizeResponse:
    """
    Accept an image file and return retrieved examples + generated Findings/Impression.
    All behavior (k, context size, models, etc.) is hardcoded.
    """
    global FAISS_INDEX, METADATA, EMBEDDER

    if FAISS_INDEX is None or EMBEDDER is None:
        raise HTTPException(status_code=503, detail="Server not ready (index or models not loaded).")

    # read uploaded image into PIL
    try:
        pil_img = Image.open(BytesIO(file)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # 1) Embed
    q_emb = EMBEDDER.embed_image_pil(pil_img)  # shape (1, D)

    # 2) Retrieve top-k
    scores, ids = retrieve_topk(FAISS_INDEX, q_emb, TOP_K)

    # 3) Map ids -> metadata
    results = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0 or idx >= len(METADATA):
            continue
        m = METADATA[idx].copy()
        m["_score"] = float(score)
        # ensure fields exist
        m.setdefault("findings", "")
        m.setdefault("impression", "")
        results.append(m)

    # 4) Compose prompt from top N context reports
    top_n = min(MAX_CONTEXT_REPORTS, len(results))
    context_reports = results[:top_n]
    prompt = assemble_prompt_from_reports(context_reports, query_projection=None)

    return prompt