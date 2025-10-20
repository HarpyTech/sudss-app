#!/usr/bin/env python3
"""
medgemma_local.py

Callable module + CLI for running MedGemma-style image+text -> text inference.

Primary function to use from other code:
    generate_from_bytes(image_bytes: bytes, prompt: str, *,
                        model_variant: str = "4b-it",
                        hf_token: Optional[str] = None,
                        device: Optional[str] = None,
                        use_quant: bool = True,
                        max_new_tokens: int = 300) -> dict

It returns a dict:
    {
      "generated_text": str,
      "raw_output": Any,
      "prompt": str
    }

The module also supports CLI usage similar to your original script.
"""

import os
import sys
from io import BytesIO
from typing import Optional, Any, Dict
from pathlib import Path

from PIL import Image
import torch
from transformers import pipeline, BitsAndBytesConfig, logging
from huggingface_hub import login as hf_login  # optional, used if token provided

# reduce transformer logs unless you want them
logging.set_verbosity_error()

# Simple pipeline cache to avoid reloading model multiple times
_PIPELINE_CACHE: Dict[str, Any] = {}


def _get_device_and_kwargs(device_arg: Optional[str], use_quant: bool):
    """Return (device_for_pipeline, model_kwargs)"""
    use_cuda = torch.cuda.is_available() and (device_arg is None or "cuda" in device_arg)
    if use_cuda:
        device_for_pipeline = 0  # pipeline accepts GPU index int
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_for_pipeline = "cpu"
        device_map = "cpu"
        torch_dtype = torch.float32

    quant_config = None
    if use_quant and use_cuda:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    return device_for_pipeline, model_kwargs


def _load_pipeline(model_variant: str = "4b-it", hf_token: Optional[str] = None,
                   device_arg: Optional[str] = None, use_quant: bool = True):
    """
    Load and cache the pipeline for the given model variant.
    model_variant -> model id constructed as 'google/medgemma-{variant}' by default.
    """
    model_id = f"google/medgemma-{model_variant}"
    cache_key = f"{model_id}|device={device_arg}|quant={use_quant}"

    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    # Optionally login (not necessary for pipeline if you pass use_auth_token)
    if hf_token:
        try:
            hf_login(hf_token)
        except Exception:
            # non-fatal: pipeline will try to use token if provided via use_auth_token
            pass

    device_for_pipeline, model_kwargs = _get_device_and_kwargs(device_arg, use_quant)

    # Task name used in your original script:
    task_name = "image-text-to-text"

    try:
        pipe = pipeline(
            task_name,
            model=model_id,
            model_kwargs=model_kwargs,
            # if you need to authenticate with a token, uncomment next line:
            use_auth_token=hf_token if hf_token else None,
            device=device_for_pipeline,
        )
    except Exception as exc:
        # Fallback: try loading without model_kwargs/device config
        try:
            pipe = pipeline(task_name, model=model_id, use_auth_token=hf_token if hf_token else None)
        except Exception as exc2:
            raise RuntimeError(f"Failed to load pipeline for {model_id}: {exc2}") from exc2

    # deterministic: prefer greedy decoding unless user overrides later
    try:
        pipe.model.generation_config.do_sample = False
    except Exception:
        pass

    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def _pil_from_bytes(image_bytes: bytes) -> Image.Image:
    bio = BytesIO(image_bytes)
    img = Image.open(bio).convert("RGB")
    return img


def _build_messages(prompt: str, image_pil: Image.Image, role_instruction: Optional[str] = None):
    role_instruction = role_instruction or "You are an expert radiologist. Provide concise Findings and Impression referencing the image and cite uncertainty when appropriate."
    messages = [
        {"role": "system", "content": [{"type": "text", "text": role_instruction}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_pil},
            ],
        },
    ]
    return messages


def generate_from_bytes(image_bytes: bytes,
                        prompt: str,
                        *,
                        model_variant: str = "4b-it",
                        hf_token: Optional[str] = None,
                        device: Optional[str] = None,
                        use_quant: bool = True,
                        max_new_tokens: int = 300,
                        role_instruction: Optional[str] = None) -> Dict[str, Any]:
    """
    Main callable function.

    Arguments:
        image_bytes: bytes of the image (PNG/JPG/etc)
        prompt: textual prompt/instruction to accompany the image
        model_variant: medgemma variant suffix, default '4b-it' -> model id 'google/medgemma-4b-it'
        hf_token: optional HF token (string). You can also set HF_TOKEN env var.
        device: optional device string, e.g., 'cuda' or 'cpu'. If None auto-detects.
        use_quant: attempt to use 4-bit quantization if CUDA + bitsandbytes available.
        max_new_tokens: generation length
        role_instruction: optional system instruction

    Returns:
        dict with keys: 'generated_text', 'raw_output', 'prompt'
    """
    # get hf token from env if not provided
    env_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = env_token

    # load pipeline (cached)
    pipe = _load_pipeline(model_variant=model_variant, hf_token=hf_token, device_arg=device, use_quant=use_quant)

    # convert bytes -> PIL image
    image_pil = _pil_from_bytes(image_bytes)

    # build messages structure expected by pipeline
    messages = _build_messages(prompt, image_pil, role_instruction=role_instruction)

    # run pipeline
    max_tokens = max_new_tokens or 300
    output = pipe(text=messages, max_new_tokens=max_tokens)

    # try to extract generated text safely (several pipeline return shapes)
    generated_text = None
    try:
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
            # common case: [{'generated_text': '...'}]
            if "generated_text" in output[0]:
                generated_text = output[0]["generated_text"]
            # nested style: output[0]['generated_text'] might itself be list/dict
            elif isinstance(output[0].get("generated_text"), list):
                # try to find last text piece
                for seg in reversed(output[0]["generated_text"]):
                    if isinstance(seg, dict) and "content" in seg:
                        generated_text = seg["content"]
                        break
                if generated_text is None:
                    generated_text = str(output[0]["generated_text"])
            elif "text" in output[0]:
                generated_text = output[0]["text"]
            else:
                # fallback to stringifying
                generated_text = str(output[0])
        else:
            generated_text = str(output)
    except Exception:
        generated_text = str(output)

    return {"generated_text": generated_text, "raw_output": output, "prompt": prompt}


# # ------------------------
# # CLI wrapper (keeps backward compatibility)
# # ------------------------
# def _cli_main():
#     import argparse

#     p = argparse.ArgumentParser(description="MedGemma local inference (callable module + CLI).")
#     group = p.add_mutually_exclusive_group(required=False)
#     group.add_argument("--image-url", type=str, help="URL (for convenience; will be downloaded)")
#     group.add_argument("--image-file", type=str, help="Local path to image file")
#     p.add_argument("--model-variant", type=str, default="4b-it")
#     p.add_argument("--use-quant", action="store_true", default=False)
#     p.add_argument("--hf-token", type=str, default=None)
#     p.add_argument("--max-new-tokens", type=int, default=300)
#     p.add_argument("--prompt", type=str, help="Text prompt to send with image.")
#     p.add_argument("--role-instruction", type=str, help="System role instruction.")
#     p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detect if omitted)")
#     args = p.parse_args()

#     # prepare image bytes (download if URL provided)
#     if args.image_file:
#         b = Path(args.image_file).read_bytes()
#     elif args.image_url:
#         import requests

#         resp = requests.get(args.image_url)
#         resp.raise_for_status()
#         b = resp.content
#     else:
#         # fallback: try default example image from Wikimedia (same as before)
#         import requests

#         url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
#         resp = requests.get(url)
#         resp.raise_for_status()
#         b = resp.content

#     prompt_text = args.prompt or "Describe this X-ray, and provide a concise Findings and Impression."
#     out = generate_from_bytes(b, prompt_text,
#                               model_variant=args.model_variant,
#                               hf_token=args.hf_token,
#                               device=args.device,
#                               use_quant=args.use_quant,
#                               max_new_tokens=args.max_new_tokens,
#                               role_instruction=args.role_instruction)
#     print("\n=== PROMPT ===\n")
#     print(out["prompt"])
#     print("\n=== GENERATED ===\n")
#     print(out["generated_text"])
#     print("\n=== RAW OUTPUT (for debugging) ===\n")
#     print(out["raw_output"])


# if __name__ == "__main__":
#     _cli_main()
