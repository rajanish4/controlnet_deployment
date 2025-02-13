import torch
import cv2
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
    def __init__(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

    def generate(self, input_image: np.ndarray, params: dict) -> (np.ndarray, np.ndarray):
        with torch.no_grad():
            # Preprocess input image
            img = resize_image(HWC3(input_image), params['image_resolution'])
            H, W, C = img.shape

            # Compute detected edges (control image)
            detected_map = self.apply_canny(img, params['low_threshold'], params['high_threshold'])
            detected_map = HWC3(detected_map)  # Detected edge image as a numpy array

            # Prepare control tensor for the model
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(params['num_samples'])], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if params['seed'] == -1:
                params['seed'] = random.randint(0, 65535)
            seed_everything(params['seed'])

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

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
                shape = (4, H // 8, W // 8)

                if config.save_memory:
                    self.model.low_vram_shift(is_diffusing=True)

                self.model.control_scales = (
                    [params['strength'] * (0.825 ** float(12 - i)) for i in range(13)]
                    if params['guess_mode'] else ([params['strength']] * 13)
                )

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

                x_samples = self.model.decode_first_stage(samples)
                x_samples = (
                    einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
                ).cpu().numpy().clip(0, 255).astype(np.uint8)
                
                # Return both the generated image and the detected edges (control image)
                return x_samples[0], detected_map
