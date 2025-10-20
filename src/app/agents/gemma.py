# app.py
import os
import sys
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
from huggingface_hub import login
from transformers import pipeline, BitsAndBytesConfig, logging

logging.set_verbosity_error()

app = FastAPI(title="MedGemma FastAPI")

# Global pipeline (loaded at startup)
PIPE = None
MODEL_ID = None


def prepare_model(model_variant: str = "4b-it", use_quant: bool = True, hf_token: Optional[str] = None):
    """
    Load the MedGemma pipeline into memory and return it.
    This mirrors the logic from your script: device selection, optional 4-bit quantization.
    """
    global MODEL_ID

    if hf_token:
        login(hf_token)

    model_id = f"google/medgemma-{model_variant}"
    MODEL_ID = model_id

    # Device selection
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = 0
        device_map = "auto"
        print("CUDA available -> will attempt GPU (device_map='auto').")
    else:
        device = "cpu"
        device_map = "cpu"
        print("No CUDA -> using CPU.")

    quant_config = None
    if use_quant and use_cuda:
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        print("4-bit quantization enabled (bitsandbytes).")
    elif use_quant and not use_cuda:
        print("Warning: 4-bit quant requested but no CUDA available; ignoring quantization.")

    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map=device_map,
    )
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    task_name = "image-text-to-text"
    try:
        pipe = pipeline(task_name, model=model_id, model_kwargs=model_kwargs)
    except Exception as e:
        # fallback: try simpler load without model_kwargs
        print("Model load failed with model_kwargs — retrying with simpler config...", file=sys.stderr)
        pipe = pipeline(task_name, model=model_id)

    # try to enforce deterministic behavior
    try:
        pipe.model.generation_config.do_sample = False
    except Exception:
        pass

    return pipe


def load_pipeline_at_startup():
    """
    Load the model once on startup. Config controlled by env vars:
      HF_TOKEN, MODEL_VARIANT, USE_QUANT (true/false)
    """
    global PIPE
    hf_token = os.environ.get("HF_TOKEN")
    model_variant = os.environ.get("MODEL_VARIANT", "4b-it")
    use_quant_env = os.environ.get("USE_QUANT", "true").lower()
    use_quant = use_quant_env not in ("0", "false", "no")

    try:
        PIPE = prepare_model(model_variant=model_variant, use_quant=use_quant, hf_token=hf_token)
        print("Model loaded and ready.")
    except Exception as e:
        print("Failed to load model during startup:", e, file=sys.stderr)
        # keep PIPE as None so endpoint returns helpful error
        PIPE = None


class InferenceResult(BaseModel):
    prompt: str
    generated_text: Optional[str]
    raw_output: Optional[str]

from PIL import Image
import base64
import io

def pil_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL image to a base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return f"data:image/{format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def convert_images_to_base64(data):
    """Recursively traverse the data structure and convert PIL.Image objects to base64 strings."""
    if isinstance(data, list):
        return [convert_images_to_base64(item) for item in data]
    elif isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key == "image" and isinstance(value, Image.Image):
                new_dict[key] = pil_image_to_base64(value)
            elif key == "image" and not isinstance(value, Image.Image):
                new_dict[key] = str(value)
            else:
                new_dict[key] = convert_images_to_base64(value)
        return new_dict
    else:
        return data


def infer(
    file: bytes,
    prompt: Optional[str] = None,
    max_new_tokens: int = 300,
    role_instruction: Optional[str] = None,
):
    """
    Accept an image file (multipart/form-data) and optional prompt and role_instruction.
    Returns JSON with generated text and raw pipeline output (stringified).
    """
    global PIPE, MODEL_ID

    hf_token = os.environ.get("HF_TOKEN")
    model_variant = os.environ.get("MODEL_VARIANT", "4b-it")
    use_quant_env = os.environ.get("USE_QUANT", "true").lower()
    use_quant = use_quant_env not in ("0", "false", "no")

    try:
        if PIPE is None:
            PIPE = prepare_model(model_variant=model_variant, use_quant=use_quant, hf_token=hf_token)
        print("Model loaded and ready.")
    except Exception as e:
        print("Failed to load model during startup:", e, file=sys.stderr)
        # keep PIPE as None so endpoint returns helpful error
        PIPE = None


    if PIPE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs / HF token.")

    # read file bytes and convert to PIL image
    try:
        image = Image.open(BytesIO(file)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read image file: {e}")

    # Build messages same shape your script used
    role_instruction = (
        role_instruction
        or "You are an expert radiologist. Explain how the summarization was defined with respect to the given X-ray."
    )
    prompt = (
        prompt
        or "Describe this X-ray, And Provide the summarization by explaining how the Findings has been captured"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": role_instruction}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    # Run pipeline (synchronous)
    try:
        output = PIPE(text=messages, max_new_tokens=max_new_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Attempt to extract generated text similar to your script
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
    
    content = dict(generated_text)

    print("Type of generated_text:", type(generated_text))
    # save to json file on current directory
    import json
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inference_output_{timestamp}.json"

    data_result = convert_images_to_base64(list(generated_text))
    try:
        summary = json.loads(data_result)
    except Exception:
        summary = {"output": data_result}   
    # summary = replace_images(generated_text)
    with open(filename, "w") as f:
        json.dump(summary, f, indent=4)
    
    # return InferenceResult(prompt=prompt, generated_text=generated_text, raw_output=str(output))
    return data_result

def replace_images(obj):
    if isinstance(obj, list):
        for item in obj:
            replace_images(item)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key == "image":
                obj[key] = "IMAGE_REPLACED"
            else:
                replace_images(value)

