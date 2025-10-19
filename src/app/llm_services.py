import os
import io
import json
from typing import Optional, List, Dict, Any

import google.generativeai as genai

# Configure the Gemini API key
# Make sure to set your GOOGLE_API_KEY as an environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize the models
summary_model = genai.GenerativeModel('gemini-2.5-pro')
scoring_model = genai.GenerativeModel('gemini-2.5-flash')


def generate_summary(image_data: bytes = None, text_data: str = None) -> str:
    """
    Generates a diagnostic summary using Gemini 1.5 Pro.
    Accepts either image or text data, or both.
    """
    prompt_parts = []
    
    base_prompt = """
    You are a clinical AI assistant. Analyze the provided medical data and generate a structured diagnostic summary.
    The summary should be in markdown format with two sections: '### Findings' and '### Impressions'.
    - Findings: Detail the objective observations from the data.
    - Impressions: Provide a concise clinical interpretation and potential diagnosis.
    Maintain professional medical terminology.
    """
    prompt_parts.append(base_prompt)

    if image_data:
        image_part = {
            "mime_type": "image/jpeg", # Assuming JPEG, adjust if needed
            "data": image_data
        }
        prompt_parts.append(image_part)

    if text_data:
        prompt_parts.append(f"\nClinical Notes/Text Input:\n{text_data}")

    if not image_data and not text_data:
        raise ValueError("Either image or text data must be provided.")

    response = summary_model.generate_content(prompt_parts)
    return response.text

def calculate_trust_score_and_regenerate(original_summary: str, feedback: str) -> tuple[float, str]:
    """
    Uses Gemini 1.5 Flash to calculate a trust score based on feedback and
    Gemini 1.5 Pro to regenerate the summary.
    """
    # --- Step 1: Calculate Trust Score using Gemini Flash ---
    scoring_prompt = f"""
    Analyze the original AI-generated medical summary and the clinician's feedback.
    Based on the severity and nature of the correction, provide a "Trust Score" between 0 and 100 for the ORIGINAL summary.
    A score of 100 means the original summary was perfect. A score of 0 means it was completely wrong.
    Consider factual errors (e.g., wrong side of the body) as major deductions.
    Consider stylistic preferences or minor additions as small deductions.
    
    Return ONLY the numerical score. For example: 85.5

    Original Summary:
    "{original_summary}"

    Clinician Feedback:
    "{feedback}"

    Trust Score:
    """
    
    score_response = scoring_model.generate_content(scoring_prompt)
    try:
        # Clean up the response to ensure it's a float
        score_text = score_response.text.strip().replace('%', '')
        trust_score = float(score_text)
    except (ValueError, TypeError):
        print(f"Could not parse score from response: '{score_response.text}'. Defaulting to 0.")
        trust_score = 0.0 # Default score if parsing fails

    # --- Step 2: Regenerate Summary using Gemini Pro ---
    regeneration_prompt = f"""
    You are a clinical AI assistant. You previously generated a medical summary which a clinician has reviewed.
    Your task is to regenerate the summary by incorporating the clinician's feedback accurately.
    Produce a new, corrected summary in the same '### Findings' and '### Impressions' markdown format.

    Original Summary:
    "{original_summary}"

    Clinician's Feedback for Correction:
    "{feedback}"

    Regenerated Summary:
    """

    regenerated_response = summary_model.generate_content(regeneration_prompt)
    regenerated_summary = regenerated_response.text

    return trust_score, regenerated_summary


# ---------------------- Additional MedSigLIP + MedGemma helpers ----------------------
def _safe_bytes_to_pil(image_data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def load_metadata_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load metadata from a jsonl file. Returns list of dicts."""
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


class MedSigLIPEmbedder:
    """Produce image embeddings compatible with MedSigLIP-like models."""
    def __init__(self, model_name: str, device: Optional[str] = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.device if isinstance(self.device, str) else str(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # load the model that produces image features; trust_remote_code may be required for med models
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def embed_pil(self, pil_image) -> Any:
        import torch
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
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
                feats = torch.tensor(feats).to(self.model.device)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            emb = feats.cpu().numpy().astype("float32")  # shape (1, D)
        return emb


def retrieve_topk_from_image_bytes(image_data: bytes,
                                   index_path: str,
                                   meta_path: str,
                                   medsiglip_model: str = "google/medsiglip-448",
                                   device: Optional[str] = None,
                                   k: int = 5) -> List[Dict[str, Any]]:
    """Embed the provided image bytes with MedSigLIP and retrieve top-k metadata from a FAISS index.

    Returns a list of metadata dicts with added keys: `_score` and `_rank`.
    """
    try:
        import faiss
    except Exception:
        raise RuntimeError("faiss is required for retrieval. Add faiss-cpu to your environment.")

    from pathlib import Path
    p_idx = Path(index_path)
    p_meta = Path(meta_path)
    if not p_idx.exists():
        raise FileNotFoundError(f"Index not found: {p_idx}")
    if not p_meta.exists():
        raise FileNotFoundError(f"Metadata not found: {p_meta}")

    metadata = load_metadata_jsonl(str(p_meta))

    pil = _safe_bytes_to_pil(image_data)
    embedder = MedSigLIPEmbedder(medsiglip_model, device=device)
    q_emb = embedder.embed_pil(pil)  # (1, D)

    idx = faiss.read_index(str(p_idx))
    k_search = min(k, idx.ntotal)
    if k_search == 0:
        return []
    scores, ids = idx.search(q_emb, k_search)
    results = []
    for rank, (i, s) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
        if i < 0 or i >= len(metadata):
            continue
        m = metadata[i].copy()
        m["_score"] = float(s)
        
        results.append(m)
    return results


def assemble_prompt_from_reports(reports: List[Dict[str, Any]], query_projection: str = None) -> str:
    ctx_lines = []
    ctx_lines.append("You are a radiology assistant. Use the example reports below as reference.")
    ctx_lines.append("")
    for i, r in enumerate(reports, start=1):
        header = f"--- uid: {r.get('uid')} | projection: {r.get('projection')} | score: {r.get('_score'):.4f} ---"
        ctx_lines.append(header)
        f = r.get("findings") or ""
        im = r.get("impression") or ""
        max_len = 800
        # if len(f) > max_len:
        #     f = f[:max_len] + " ... [truncated]"
        # if len(im) > max_len:
        #     im = im[:max_len] + " ... [truncated]"
        ctx_lines.append("Findings: " + f)
        ctx_lines.append("Impression: " + im)
        ctx_lines.append("")
    ctx_lines.append("Now given the context for chest x-ray image")
    if query_projection:
        ctx_lines.append(f"Projection: {query_projection}")
    ctx_lines.append("Write two sections clearly labeled 'Findings:' and 'Impression:' — be concise and mention uncertainty where appropriate.")
    ctx_lines.append("")
    return "\n".join(ctx_lines)


def medgemma_generate_with_image_and_context(image_data: bytes,
                                             context_reports: List[Dict[str, Any]],
                                             model_variant: str = "4b-it",
                                             use_quant: bool = True,
                                             hf_token: Optional[str] = None,
                                             max_new_tokens: int = 300) -> str:
    """Run MedGemma local pipeline similar to `test.py`.

    image_data: raw bytes of the uploaded image
    context_reports: list of metadata dicts to include in the prompt
    Returns generated text string.
    """
    try:
        from transformers import pipeline, BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError("transformers is required for MedGemma generation. Add transformers and huggingface_hub to your environment.")

    from huggingface_hub import login
    from PIL import Image

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)

    model_id = f"google/medgemma-{model_variant}"

    # device selection
    import torch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = 0
        device_map = "auto"
    else:
        device = "cpu"
        device_map = "cpu"

    quant_config = None
    if use_quant and use_cuda:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)

    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map=device_map,
    )
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    # build prompt/messages
    prompt_text = assemble_prompt_from_reports(context_reports)

    pil = _safe_bytes_to_pil(image_data)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": pil}]},
    ]

    # try to load pipeline
    try:
        pipe = pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs)
    except Exception:
        # fallback: try simpler load
        pipe = pipeline("image-text-to-text", model=model_id)

    # attempt deterministic generation
    try:
        pipe.model.generation_config.do_sample = False
    except Exception:
        pass

    output = pipe(text=messages, max_new_tokens=max_new_tokens)

    # extract generated text safely
    generated_text = None
    try:
        if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
            generated_text = output[0]["generated_text"]
        elif isinstance(output, list) and len(output) > 0 and isinstance(output[0].get("generated_text"), list):
            generated_text = output[0]["generated_text"][-1].get("content")
        else:
            generated_text = str(output)
    except Exception:
        generated_text = str(output)
    

    print("MedGemma generated text:", generated_text)

    # remove type image info if present in the dictionary considering the generated text structure is same as message and return the the content of role as assistant
    final_report = dict(generated_text)

    # remove the key 'type': 'image' if present
    if isinstance(final_report, dict) and "type" in final_report and final_report["type"] == "image":
        del final_report["type"]

    return final_report


def retrieve_then_generate(image_data: bytes,
                           index_path: str,
                           meta_path: str,
                           medsiglip_model: str = "google/medsiglip-448",
                           medgemma_variant: str = "4b-it",
                           k: int = 5,
                           hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve top-k similar reports using MedSigLIP and FAISS, then run MedGemma with those reports as context.

    Returns dict with keys: retrieved (list), generated (str), prompt (str)
    """
    retrieved = retrieve_topk_from_image_bytes(image_data, index_path, meta_path, medsiglip_model, k=k)
    prompt = assemble_prompt_from_reports(retrieved)
    generated = medgemma_generate_with_image_and_context(image_data, retrieved, model_variant=medgemma_variant, hf_token=hf_token)
    return {"retrieved": retrieved, "prompt": prompt, "generated": generated}


def create_langchain_tools_if_available():
    """If `langchain` is installed, return two Tool objects wrapping retrieval+gen functions.
    This is optional and will silently return None if langchain isn't available.
    """
    try:
        from langchain.tools import tool
        from langchain.tools import BaseTool
    except Exception:
        return None

    # Note: Using simple function wrappers as tools. Consumers can integrate into an Agent as needed.
    def retrieval_tool(image_bytes: bytes, index_path: str, meta_path: str, k: int = 5):
        return retrieve_topk_from_image_bytes(image_bytes, index_path, meta_path, k=k)

    def medgemma_tool(image_bytes: bytes, context_reports: List[Dict[str, Any]]):
        return medgemma_generate_with_image_and_context(image_bytes, context_reports)

    return {"retrieval_tool": retrieval_tool, "medgemma_tool": medgemma_tool}
