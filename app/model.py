import torch
import einops
import numpy as np
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import config

class ControlNetModel:
    """
    ControlNetModel is a wrapper for the ControlNet model, responsible for generating synthetic images.
    
    This class handles:
      - Loading the model configuration and weights.
      - Preprocessing the input image.
      - Detecting edges with a Canny detector.
      - Generating synthetic images using a DDIM sampler.
      
    Attributes:
      apply_canny: Instance of CannyDetector for edge detection.
      model: The loaded ControlNet model.
      ddim_sampler: Sampler used to generate images from latent representations.
    """
    def __init__(self):
        """
        Initialize the ControlNetModel by setting up the Canny detector, loading the model configuration
        and weights, and initializing the DDIM sampler.
        
        Steps:
          1. Instantiate the Canny detector.
          2. Create the model from the YAML configuration file.
          3. Load the model state dictionary from file and move the model to GPU.
          4. Initialize the DDIMSampler for sampling latent representations.
        """
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def generate(self, input_image: np.ndarray, params: dict) -> (np.ndarray, np.ndarray):
        """
        Generate synthetic images and the corresponding control image (edge map) based on the given input.
        
        Process:
          1. Preprocess the input image: convert to HWC3 format and resize based on 'image_resolution'.
          2. Detect edges using the Canny detector with provided 'low_threshold' and 'high_threshold'.
          3. Prepare a control tensor from the detected edge map.
          4. Set the random seed for reproducibility based on the 'seed' parameter.
          5. Optionally switch the model to a memory saving mode if enabled via config.
          6. Define conditioning inputs using 'prompt' and 'a_prompt' for positive guidance,
             and 'n_prompt' for negative guidance.
          7. Run the DDIM sampler with the specified number of steps ('ddim_steps'), sampling parameters,
             and latent shape derived from the preprocessed image.
          8. Decode the latent representation to obtain the final generated images.
        
        Parameters:
          input_image (np.ndarray): The input image as a numpy array.
          params (dict): Dictionary containing generation parameters. Expected keys include:
                         'image_resolution', 'low_threshold', 'high_threshold', 'num_samples',
                         'seed', 'prompt', 'a_prompt', 'n_prompt', 'ddim_steps', 'guess_mode',
                         'strength', 'scale', and 'eta'.
        
        Returns:
          tuple:
            - Generated images as a numpy array of shape (num_samples, height, width, channels).
            - Detected edge map (control image) as a numpy array.
        """
        with torch.no_grad():
            # Preprocess input image
            img = resize_image(HWC3(input_image), params['image_resolution'])
            H, W, C = img.shape

            # Compute detected edges (control image)
            detected_map = self.apply_canny(img, params['low_threshold'], params['high_threshold'])
            detected_map = HWC3(detected_map)  # Detected edge image as a numpy array

            # Prepare control tensor from detected edges. Normalize the tensor and rearrange dimensions.
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(params['num_samples'])], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            # Set seed for reproducibility.
            if params['seed'] == -1:
                params['seed'] = random.randint(0, 65535)
            seed_everything(params['seed'])

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # Generate conditioning and unconditional conditioning inputs.
            with torch.cuda.amp.autocast(enabled=True):
                cond = {
                    "c_concat": [control],
                    "c_crossattn": [self.model.get_learned_conditioning(
                        [params['prompt'] + ", " + params['a_prompt']] * params['num_samples']
                    )]
                }
                un_cond = {
                    "c_concat": None if params['guess_mode'] else [control],
                    "c_crossattn": [self.model.get_learned_conditioning(
                        [params['n_prompt']] * params['num_samples']
                    )]
                }
                # Define latent shape based on the input image dimensions.
                shape = (4, H // 8, W // 8)

                if config.save_memory:
                    self.model.low_vram_shift(is_diffusing=True)

                # Set up control scales for conditioning.
                self.model.control_scales = (
                    [params['strength'] * (0.825 ** float(12 - i)) for i in range(13)]
                    if params['guess_mode'] else ([params['strength']] * 13)
                )

                # Run the DDIM sampler to generate samples.
                samples, _ = self.ddim_sampler.sample(
                    params['ddim_steps'],
                    params['num_samples'],
                    shape,
                    cond,
                    verbose=False,
                    eta=params['eta'],
                    unconditional_guidance_scale=params['scale'],
                    unconditional_conditioning=un_cond
                )

                if config.save_memory:
                    self.model.low_vram_shift(is_diffusing=False)

                # Decode the sampled latent representations to obtain image data.
                x_samples = self.model.decode_first_stage(samples)
                x_samples = (
                    einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
                ).cpu().numpy().clip(0, 255).astype(np.uint8)

                # Return all generated samples and the detected edge map.
                return x_samples, detected_map
