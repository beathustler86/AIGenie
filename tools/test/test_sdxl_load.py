from diffusers import StableDiffusionXLPipeline, StableDiffusionXLRefinerPipeline
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model
base_model_path = "F:/SoftwareDevelopment/AI Models Image/-AI_Models_Image/sd_xl_base_1.0.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(base_model_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Load refiner model
refiner_model_path = "F:/SoftwareDevelopment/AI Models Image/-AI_Models_Image/sd_xl_refiner_1.0.safetensors"
refiner = StableDiffusionXLRefinerPipeline.from_single_file(refiner_model_path, torch_dtype=torch.float16)
refiner = refiner.to(device)

print("✅ SDXL base and refiner loaded successfully.")