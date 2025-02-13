import cv2
import os
import numpy as np
import json
import io
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional

from .model import ControlNetModel
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ControlNet Demo",
    description="This API allows generation of synthetic images using the ControlNet model via REST endpoints."
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = ControlNetModel()

def get_inputs_dir():
    """Return the base inputs directory depending on environment."""
    if os.path.exists("/code/inputs"):
        return "/code/inputs"
    return str(Path.cwd() / "inputs")

def get_outputs_dir():
    """Return the base outputs directory depending on environment."""
    if os.path.exists("/code/outputs"):
        return "/code/outputs"
    return str(Path.cwd() / "outputs")

def concatenate_images(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Concatenate two images side by side."""
    h = max(img1.height, img2.height)
    w = img1.width + img2.width
    new_img = Image.new("RGB", (w, h))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img

class GenerationParams(BaseModel):
    prompt: str
    a_prompt: str = 'good quality'
    n_prompt: str = 'animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    num_samples: int = 1
    image_resolution: int = 128
    ddim_steps: int = 10
    guess_mode: bool = False
    strength: float = 1.0
    scale: float = 9.0
    seed: int = 1
    eta: float = 0.0
    low_threshold: int = 50
    high_threshold: int = 100

@app.post("/generate", summary="Generate image using direct parameters")
async def generate_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    a_prompt: str = Form("good quality"),
    n_prompt: str = Form("animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
    num_samples: int = Form(1),
    image_resolution: int = Form(128),
    ddim_steps: int = Form(10),
    guess_mode: bool = Form(False),
    strength: float = Form(1.0),
    scale: float = Form(9.0),
    seed: int = Form(1),
    eta: float = Form(0.0),
    low_threshold: int = Form(50),
    high_threshold: int = Form(100)
):
    """
    Generate a synthetic image using an uploaded image and specified parameters.
    Returns a concatenated image (control + generated).
    """
    try:
        params = GenerationParams(
            prompt=prompt,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            num_samples=num_samples,
            image_resolution=image_resolution,
            ddim_steps=ddim_steps,
            guess_mode=guess_mode,
            strength=strength,
            scale=scale,
            seed=seed,
            eta=eta,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        generated_np, control_np = model.generate(img, params.dict())
        generated_img = Image.fromarray(generated_np)
        control_img = Image.fromarray(control_np)
        base_filename, _ = os.path.splitext(file.filename)
        timestamp = datetime.now().strftime('%d_%m_%y_%H_%M_%S')
        output_filename = f"{base_filename}_{timestamp}.png"
        output_dir = get_outputs_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        generated_img.save(output_path)
        concat_img = concatenate_images(control_img, generated_img)
        img_byte_arr = io.BytesIO()
        concat_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        logger.info("Image generated successfully, saved to %s", output_path)
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("Error during image generation: %s", str(e))
        raise HTTPException(status_code=500, detail="Image generation failed.")

@app.post("/generate_from_config", summary="Generate image using a config file")
async def generate_from_config(config_file: str = Form(...)):
    """
    Generate a synthetic image using a JSON configuration file.
    The config file (located in inputs/configs/) should define 'image_path' and 'params'.
    """
    try:
        inputs_dir = get_inputs_dir()
        config_path = os.path.join(inputs_dir, "configs", config_file)
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail="Configuration file not found.")
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        params = GenerationParams(**cfg['params'])
        image_path = os.path.join(inputs_dir, cfg['image_path'])
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Input image not found.")
        img = cv2.imread(image_path)
        generated_np, control_np = model.generate(img, params.dict())
        generated_img = Image.fromarray(generated_np)
        control_img = Image.fromarray(control_np)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime('%d_%m_%y_%H_%M_%S')
        output_filename = f"{base_filename}_{timestamp}.png"
        output_dir = get_outputs_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        generated_img.save(output_path)
        concat_img = concatenate_images(control_img, generated_img)
        img_byte_arr = io.BytesIO()
        concat_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        logger.info("Image generated from config successfully, saved to %s", output_path)
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("Error during config-based image generation: %s", str(e))
        raise HTTPException(status_code=500, detail="Config-based image generation failed.")
