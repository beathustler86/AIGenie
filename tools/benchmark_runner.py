from diffusers import StableDiffusionXLPipeline
import torch, time

model_path = "models/text_to_image/sdxl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

prompt = "A photorealistic portrait of a woman in a futuristic city"
start = time.time()
image = pipe(prompt=prompt, num_inference_steps=20, width=1024, height=1024).images[0]
print(f"Generation time: {time.time() - start:.2f} seconds")
image.save("outputs/images/sdxl_benchmark.png")