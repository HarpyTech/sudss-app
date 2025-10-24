"""
Generate MedSigLIP image + text embeddings from a CSV of Chest X‑ray records and store them in a FAISS index.

Outputs:
 - faiss index file: <output_dir>/images.index
 - numpy file with image embeddings (optional): <output_dir>/image_embs.npy
 - JSONL metadata file: <output_dir>/metadata.jsonl  (one JSON object per indexed vector)
 - (optional) text embeddings saved to <output_dir>/text_embs.npy and text.index if enabled

Usage:
    python generate_med_siglip_embeddings.py \
        --csv data/cxr_reports.csv \
        --image-col image_path \
        --uid-col uid \
        --output-dir ./out_embeddings \
        --model-name google/medsiglip-448 \
        --batch-size 16

Dependencies:
    pip install torch torchvision transformers accelerate safetensors huggingface_hub pillow pandas faiss-cpu tqdm
    (replace faiss-cpu with faiss-gpu if you have GPU and want GPU faiss)

Notes:
 - This script tries to be robust to different HF model output shapes by checking for common methods
   like `get_image_features` / `get_text_features` or attributes such as `image_embeds`.
 - The script L2-normalizes embeddings and builds a FAISS IndexFlatIP (cosine via normalized vectors).
 - Keep PHI and compliance considerations in mind before storing or transferring report text.

"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel

import faiss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV file containing dataset")
    p.add_argument("--test-size", type=int, default=None, help="If set, only process this many rows (for testing)")
    p.add_argument("--image-col", default="image_path", help="CSV column that points to image file")
    p.add_argument("--uid-col", default="uid", help="CSV column with unique id")
    p.add_argument("--text-cols", nargs="*", default=["Problems", "indication", "comparison", "findings", "impression", "MeSH"],
                   help="Columns to concatenate for report text embedding (order preserved). Use none to skip text embeddings")
    p.add_argument("--model-name", default="google/medsiglip-448", help="Hugging Face model repo (MedSigLIP family recommended)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", default="./embeddings_out")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detected if not set)")
    p.add_argument("--skip-text", action="store_true", help="Skip computing text embeddings")
    p.add_argument("--max-text-tokens", type=int, default=128)
    return p.parse_args()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_dataframe(csv_path: str, test_size: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if test_size is not None and test_size > 0:
        df = df.head(test_size)
    return df


def get_device(args_device: str = None) -> torch.device:
    if args_device is not None:
        return torch.device(args_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedder:
    def __init__(self, model_name: str, device: torch.device, max_text_tokens: int = 128):
        self.device = device
        print(f"Loading model {model_name} to device {device} ...")
        # Image processor/tokenizer
        # AutoImageProcessor for images, AutoTokenizer for text
        self.img_processor = AutoImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Load model (trust_remote_code True because many Med models provide custom code)
        # Be cautious in production when using trust_remote_code
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

        self.max_text_tokens = max_text_tokens

    def embed_images(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Batch embed a list of PIL images and return a numpy array (N, D)"""
        # Preprocess images using the processor
        inputs = self.img_processor(images=pil_images, return_tensors="pt")
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Try common model helpers
            try:
                # Many multi-modal repos expose get_image_features
                image_feats = self.model.get_image_features(**inputs)
            except Exception:
                out = self.model(**inputs)
                # Try to extract in order of common attribute names
                image_feats = None
                if hasattr(out, "image_embeds"):
                    image_feats = out.image_embeds
                elif hasattr(out, "pooler_output"):
                    image_feats = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    image_feats = out.last_hidden_state[:, 0, :]
                else:
                    # Fallback: attempt to use entire output tensor
                    # This might fail for some models — in that case the user should adapt
                    image_feats = out

            # Ensure tensor
            if isinstance(image_feats, torch.Tensor):
                emb = image_feats
            else:
                emb = torch.tensor(np.asarray(image_feats)).to(self.device)

            # L2 normalize
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            emb_np = emb.cpu().numpy()
        return emb_np

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch embed texts and return numpy array (N, D)"""
        toks = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_text_tokens, return_tensors="pt")
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            try:
                text_feats = self.model.get_text_features(**toks)
            except Exception:
                out = self.model(**toks)
                text_feats = None
                if hasattr(out, "text_embeds"):
                    text_feats = out.text_embeds
                elif hasattr(out, "pooler_output"):
                    text_feats = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    text_feats = out.last_hidden_state[:, 0, :]
                else:
                    text_feats = out

            if isinstance(text_feats, torch.Tensor):
                emb = text_feats
            else:
                emb = torch.tensor(np.asarray(text_feats)).to(self.device)

            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            emb_np = emb.cpu().numpy()
        return emb_np


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS IndexFlatIP and add embeddings (assumes embeddings are L2-normalized)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_metadata_jsonl(metadata: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for obj in metadata:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def print_agrs(args):
    print("Arguments:")
    for arg, val in vars(args).items():
        print(f"  {arg}: {val}")

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = get_device(args.device)

    print_agrs(args)

    df = load_dataframe(args.csv, args.test_size if hasattr(args, 'test_size') else None)
    print(f"Loaded CSV with {len(df)} rows")

    embedder = Embedder(args.model_name, device, max_text_tokens=args.max_text_tokens)

    image_paths = df[args.image_col].fillna("").tolist()
    uids = df[args.uid_col].tolist()

    # Prepare text corpus if needed
    compute_text = not args.skip_text and len(args.text_cols) > 0
    if compute_text:
        print("Preparing text corpus by concatenating columns:", args.text_cols)
        def concat_row_text(r):
            parts = []
            for c in args.text_cols:
                if c in r and pd.notna(r[c]):
                    parts.append(str(r[c]))
            return " \n ".join(parts)
        texts = [concat_row_text(row) for _, row in df.iterrows()]
    else:
        texts = [""] * len(df)

    # Iterate in batches, compute embeddings. We'll build arrays then index.
    img_embs_list = []
    text_embs_list = [] if compute_text else None
    metadata = []

    batch_size = args.batch_size
    n = len(df)

    for i in tqdm(range(0, n, batch_size), desc="Batches"):
        batch_slice = slice(i, min(i + batch_size, n))
        batch_paths = image_paths[batch_slice]
        batch_uids = uids[batch_slice]
        batch_texts = texts[batch_slice] if compute_text else None

        # Load PIL images for those with valid paths; for missing, create a dummy blank image to keep alignment
        pil_images = []
        for pth in batch_paths:
            if isinstance(pth, str) and pth and os.path.exists(pth):
                try:
                    im = Image.open(pth).convert("RGB")
                except Exception as e:
                    print(f"Warning: failed to open image {pth}: {e}")
                    im = Image.new("RGB", (448, 448), color=(0, 0, 0))
            else:
                # fallback blank image
                im = Image.new("RGB", (448, 448), color=(0, 0, 0))
            pil_images.append(im)

        # Embed images
        try:
            batch_img_embs = embedder.embed_images(pil_images)
        except Exception as e:
            print(f"Error embedding images in batch starting at {i}: {e}")
            raise

        img_embs_list.append(batch_img_embs)

        # Embed texts if required
        if compute_text:
            try:
                batch_text_embs = embedder.embed_texts(batch_texts)
            except Exception as e:
                print(f"Error embedding texts in batch starting at {i}: {e}")
                # fallback to zeros (will be normalized later) — but better to raise in production
                batch_text_embs = np.zeros((len(batch_texts), batch_img_embs.shape[1]), dtype=np.float32)
            text_embs_list.append(batch_text_embs)

        # Collect metadata objects for each item in batch
        for idx_in_batch, uid in enumerate(batch_uids):
            row_idx = i + idx_in_batch
            row = df.iloc[row_idx].to_dict()
            # store minimal metadata; avoid storing raw PHI unless intended
            m = {
                "uid": row.get(args.uid_col),
                "image_path": row.get(args.image_col),
                # include projection, maybe MeSH tags for filtering
                "projection": row.get("projection"),
                "MeSH": row.get("MeSH"),
                # include brief findings/impression for quick preview (careful re PHI)
                "findings": row.get("findings"),
                "impression": row.get("impression"),
            }
            metadata.append(m)

    # Stack embeddings
    all_img_embs = np.vstack(img_embs_list).astype("float32")
    print("Image embeddings shape:", all_img_embs.shape)

    if compute_text:
        all_text_embs = np.vstack(text_embs_list).astype("float32")
        print("Text embeddings shape:", all_text_embs.shape)
    else:
        all_text_embs = None

    # Build FAISS index (images)
    print("Building FAISS index for image embeddings ...")
    # Ensure embeddings are L2-normalized (they should be from embedder but double-check)
    faiss.normalize_L2(all_img_embs)
    img_index = build_faiss_index(all_img_embs)

    index_path = os.path.join(args.output_dir, "images.index")
    print(f"Saving FAISS index to {index_path} ...")
    faiss.write_index(img_index, index_path)

    # Save embeddings as numpy (optional)
    emb_np_path = os.path.join(args.output_dir, "image_embs.npy")
    np.save(emb_np_path, all_img_embs)

    # Save metadata JSONL
    meta_path = os.path.join(args.output_dir, "metadata.jsonl")
    save_metadata_jsonl(metadata, meta_path)
    print(f"Saved metadata to {meta_path}")

    # Optionally save text embeddings and text index
    if compute_text and all_text_embs is not None:
        faiss.normalize_L2(all_text_embs)
        text_index = build_faiss_index(all_text_embs)
        text_index_path = os.path.join(args.output_dir, "text.index")
        faiss.write_index(text_index, text_index_path)
        np.save(os.path.join(args.output_dir, "text_embs.npy"), all_text_embs)
        print(f"Saved text index and embeddings to {args.output_dir}")

    print("All done.")


if __name__ == "__main__":
    main()
