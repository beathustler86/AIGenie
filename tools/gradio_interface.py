import gradio as gr
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

# Paths to your models
base_path = "models/text_to_image/sdxl-base-1.0"
refiner_path = "models/text_to_image/sdxl-refiner-1.0"

# Load pipelines once at startup
base_pipe = StableDiffusionXLPipeline.from_pretrained(
    base_path,
    torch_dtype=torch.float16
).to("cuda")

refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_path,
    torch_dtype=torch.float16
).to("cuda")

def sdxl_generate(prompt, steps_base, steps_refiner, use_refiner, width, height):
    # Generate base latent
    base_result = base_pipe(
        prompt=prompt,
        num_inference_steps=steps_base,
        output_type="latent",
        width=width,
        height=height
    )
    latent = base_result.images[0]
    if use_refiner:
        # Refine the latent with the refiner pipeline
        refined_result = refiner_pipe(
            prompt=prompt,
            image=latent,
            num_inference_steps=steps_refiner
        )
        img = refined_result.images[0]
    else:
        # Decode the latent to an image using the base pipeline
        img = base_pipe.decode_latents(latent.unsqueeze(0))[0]
    return img

with gr.Blocks() as demo:
    gr.Markdown("# SDXL Base + Refiner Web UI")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="A photorealistic portrait of a woman in a futuristic city")
    with gr.Row():
        steps_base = gr.Slider(10, 50, value=15, step=1, label="Base Steps")
        steps_refiner = gr.Slider(5, 30, value=7, step=1, label="Refiner Steps")
        use_refiner = gr.Checkbox(label="Use Refiner", value=True)
    with gr.Row():
        width = gr.Slider(512, 1024, value=768, step=64, label="Width")
        height = gr.Slider(512, 1024, value=768, step=64, label="Height")
    with gr.Row():
        btn = gr.Button("Generate")
    with gr.Row():
        output = gr.Image(label="Generated Image")

    btn.click(
        fn=sdxl_generate,
        inputs=[prompt, steps_base, steps_refiner, use_refiner, width, height],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()