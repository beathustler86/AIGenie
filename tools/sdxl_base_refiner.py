from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

# Paths to your models
base_path = "models/text_to_image/sdxl-base-1.0"
refiner_path = "models/text_to_image/sdxl-refiner-1.0"

# Load base pipeline
base = StableDiffusionXLPipeline.from_pretrained(
    base_path,
    torch_dtype=torch.float16
).to("cuda")

prompt = "A photorealistic portrait of a woman in a futuristic city"

# Generate latent image with base
base_image = base(
    prompt=prompt,
    num_inference_steps=20,
    output_type="latent"
).images[0]

# Load refiner pipeline (img2img)
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_path,
    torch_dtype=torch.float16
).to("cuda")

# Refine the image
refined_image = refiner(
    prompt=prompt,
    image=base_image,
    num_inference_steps=10
).images[0]

refined_image.save("outputs/images/sdxl_refined_result.png")
print("Image saved to outputs/images/sdxl_refined_result.png")