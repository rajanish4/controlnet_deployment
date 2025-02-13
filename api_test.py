import requests
import json
from pathlib import Path

class ControlNetAPITester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.default_params = {
            'prompt': 'MRI brain scan',
            'a_prompt': 'good quality',
            'n_prompt': 'animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
            'num_samples': 1,
            'image_resolution': 256,
            'ddim_steps': 5,
            'guess_mode': 'false',
            'strength': 1.0,
            'scale': 9.0,
            'seed': 1,
            'eta': 0.0,
            'low_threshold': 50,
            'high_threshold': 100
        }

    def test_generate_endpoint(self, image_path, output_path="output_direct.png"):
        """Test the /generate endpoint with direct parameters"""
        url = f"{self.base_url}/generate"
        
        with open(image_path, 'rb') as f:
            files = {'file': ('image.jpg', f, 'image/jpeg')}
            response = requests.post(url, files=files, data=self.default_params)

        self._handle_response(response, output_path)

    def test_generate_from_config(self, config_file, output_path="output_config.png"):
        """Test the /generate_from_config endpoint using a config file"""
        url = f"{self.base_url}/generate_from_config"
        response = requests.post(url, data={'config_file': config_file})
        self._handle_response(response, output_path)

    def _handle_response(self, response, output_path):
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Image generated successfully! Saved to {output_path}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

def main():
    tester = ControlNetAPITester()
    
    # Test direct generation
    tester.test_generate_endpoint('inputs/images/mri_brain.jpg')
    
    # Test config-based generation
    tester.test_generate_from_config('parameters.json')

if __name__ == "__main__":
    main() 