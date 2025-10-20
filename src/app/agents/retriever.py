import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
import faiss

def get_device(args_device: str = None) -> torch.device:
    if args_device:
        return torch.device(args_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metadata_jsonl(path: str=None) -> List[Dict[str, Any]]:
    meta = []
    path = path or "../embeddings/metadata.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


class MedSigLIPEmbedder:
    """Small helper to produce an image embedding for a query image using MedSigLIP-like model."""
    def __init__(self, device: torch.device):
        self.device = device
        model_name = "google/medsiglip-448"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # load the model that produces image features; trust_remote_code may be required for med models
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    def embed_image(self, image_path: bytes) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
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


def retrieve_topk(q_emb: np.ndarray, k: int):
    index_path = "../embeddings/images.index"
    idx = faiss.read_index(index_path)
    k_search = min(k, idx.ntotal)
    if k_search == 0:
        return [], []
    scores, ids = idx.search(q_emb, k_search)
    return scores[0].tolist(), ids[0].tolist()


def assemble_prompt_from_reports(reports: List[Dict[str, Any]], query_projection: str = None) -> str:
    """
    Build a single prompt string that concatenates the retrieved reports as context,
    then asks the model to generate Findings and Impression for the query image.
    """
    ctx_lines = []
    ctx_lines.append("You are a radiology assistant. Use the example reports below as reference.")
    ctx_lines.append("")  # blank line
    ctx_lines.append("Here are some example chest x-ray reports for the CONTEXT:")
    ctx_lines.append("")  # blank line
    for i, r in enumerate(reports, start=1):
        header = f"--- uid: {r.get('uid')} | projection: {r.get('projection')} | score: {r.get('_score'):.4f} ---"
        ctx_lines.append(header)
        # keep Findings and Impression short but present
        f = r.get("findings") or ""
        im = r.get("impression") or ""
        # truncate very long fields to avoid overly long context
        # max_len = 800
        # if len(f) > max_len:
        #     f = f[:max_len] + " ... [truncated]"
        # if len(im) > max_len:
        #     im = im[:max_len] + " ... [truncated]"
        ctx_lines.append("Findings: " + f)
        ctx_lines.append("Impression: " + im)
        ctx_lines.append("")  # blank line between examples

    # ctx_lines.append("Now given the query chest x-ray image (use the same style as the examples),")
    # if query_projection:
    #     ctx_lines.append(f"Projection: {query_projection}")
    # ctx_lines.append("Write two sections clearly labeled 'Findings:' and 'Impression:' — be concise and mention uncertainty where appropriate.")
    ctx_lines.append("")  # final blank
    prompt = "\n".join(ctx_lines)
    return prompt


def generate_with_text_model(prompt: str, model_name: str, device: torch.device, max_new_tokens: int = 256) -> str:
    """
    Simple example using an AutoModelForCausalLM-compatible text model.
    Replace this with your MedGemma text-serving code or remote API if needed.
    """
    # NOTE: If using a very large model, you may want to call a remote endpoint instead.
    print(f"Loading generation model {model_name} on {device} (this may take time)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def retrieve_topfive(query_image: bytes):
    device = get_device()
    print("Using device:", device)

    # # basic checks
    # if not idx_p.exists():
    #     raise FileNotFoundError("Index not found: " + str(idx_p))
    # if not meta_p.exists():
    #     raise FileNotFoundError("Metadata not found: " + str(meta_p))

    # load metadata
    metadata = load_metadata_jsonl()
    print(f"Loaded {len(metadata)} metadata records")

    # embed query image
    embder = MedSigLIPEmbedder(device)
    q_emb = embder.embed_image(query_image)  # shape (1, D)

    # retrieve top-k
    scores, ids = retrieve_topk(q_emb, 5)
    print("Retrieved ids:", ids)
    # build results list by mapping ids to metadata
    results = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx].copy()
        m["_score"] = float(score)
        results.append(m)

    # Optionally filter by projection similarity (uncomment if you want)
    query_proj = None
    # If your metadata keeps projection, you can set query_proj from query or pass it in
    # query_proj = "AP"

    # Compose prompt using top-N retrieved reports
    top_n = min(5, len(results))
    context_reports = results[:top_n]
    prompt = assemble_prompt_from_reports(context_reports, query_projection=query_proj)

    print("\n--- Prompt (truncated 1000 chars) ---\n")
    print(prompt)
    print("...\n--- end prompt preview ---\n")

    return prompt, results

