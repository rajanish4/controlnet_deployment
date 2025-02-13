import cv2
import os
import numpy as np
import json
import io
from pathlib import Path

from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional

from .model import ControlNetModel
from transformers import logging
logging.set_verbosity_error()

# Create FastAPI instance with a title and description.
app = FastAPI(title="ControlNet Demo")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# Pydantic model for validating and documenting the generation parameters.
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

@app.on_event("startup")
async def startup_event():
    global model
    model = ControlNetModel()

def concatenate_images(left_img: Image.Image, right_img: Image.Image) -> Image.Image:
    """
    Concatenate two images side by side.
    Creates a new image with width equal to the sum of widths and height equal to the maximum of both images.
    """
    w1, h1 = left_img.size
    w2, h2 = right_img.size
    new_width = w1 + w2
    new_height = max(h1, h2)
    concatenated = Image.new("RGB", (new_width, new_height))
    concatenated.paste(left_img, (0, 0))
    concatenated.paste(right_img, (w1, 0))
    return concatenated

def get_inputs_dir():
    """
    Return the base inputs directory depending on the environment.
    If '/code/inputs' exists (e.g., when running in Docker), use it;
    otherwise, use the local 'inputs' directory.
    """
    if os.path.exists("/code/inputs"):
        return "/code/inputs"
    else:
        return str(Path.cwd() / "inputs")

def get_outputs_dir():
    """
    Return the base outputs directory depending on the environment.
    If '/code/outputs' exists (e.g., when running in Docker), use it;
    otherwise, use the local 'outputs' directory.
    """
    if os.path.exists("/code/outputs"):
        return "/code/outputs"
    else:
        return str(Path.cwd() / "outputs")

@app.post("/generate")
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
    Endpoint to generate a synthetic image using an uploaded input image and given generation parameters.
    
    The endpoint returns a concatenated image that shows both the control (edge-detected) image and the generated image.
    """
    # Assemble the GenerationParams model from form fields
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
    
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Call the model to get both generated and control images
    generated_np, control_np = model.generate(img, params.dict())
    
    # Convert numpy arrays to PIL Images
    generated_img = Image.fromarray(generated_np)
    control_img = Image.fromarray(control_np)
    
    # Create unique filename with timestamp
    base_filename, _ = os.path.splitext(file.filename)
    timestamp = datetime.now().strftime('%d_%m_%y_%H_%M_%S')
    output_filename = f"{base_filename}_{timestamp}.png"
    
    # Use dynamic output directory
    output_dir = get_outputs_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Save generated image
    generated_img.save(output_path)
    
    # Concatenate control and generated images
    concat_img = concatenate_images(control_img, generated_img)
    
    # Convert to bytes for response
    img_byte_arr = io.BytesIO()
    concat_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.post("/generate_from_config")
async def generate_from_config(config_file: str = Form(...)):
    """
    Endpoint to generate a synthetic image using a JSON configuration file.
    
    The specified config file (located in inputs/configs/) must define an 'image_path' and corresponding 'params'.
    """
    # Dynamically get the base inputs directory.
    inputs_dir = get_inputs_dir()
    config_path = os.path.join(inputs_dir, "configs", config_file)
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Validate parameters using GenerationParams model
    params = GenerationParams(**cfg['params'])
    
    # Read input image using the dynamic inputs directory
    image_path = os.path.join(inputs_dir, cfg['image_path'])
    img = cv2.imread(image_path)
    
    # Generate images
    generated_np, control_np = model.generate(img, params.dict())
    
    # Convert to PIL Images
    generated_img = Image.fromarray(generated_np)
    control_img = Image.fromarray(control_np)
    
    # Create unique filename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime('%d_%m_%y_%H_%M_%S')
    output_filename = f"{base_filename}_{timestamp}.png"
    
    # Use dynamic output directory
    output_dir = get_outputs_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Save generated image
    generated_img.save(output_path)
    
    # Concatenate images
    concat_img = concatenate_images(control_img, generated_img)
    
    # Convert to bytes for response
    img_byte_arr = io.BytesIO()
    concat_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
