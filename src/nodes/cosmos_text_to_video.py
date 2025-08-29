import os
import torch
from PIL import Image
import sys  # ✅ Must come before using sys.modules

# 🧠 Redirect legacy import path for Cosmos checkpoint compatibility
import nodes.cosmos.cosmos_tokenizer.utils as actual_utils
sys.modules["nodes.cosmos.utils"] = actual_utils

from nodes.ops import operations

pos_emb_cls = "sincos"



class CosmosTextToVideo:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"text_prompt": ("STRING", {"multiline": True}),
				"frame_count": ("INT", {"default": 24, "min": 8, "max": 128}),
				"width": ("INT", {"default": 1280, "min": 256, "max": 1920}),
				"height": ("INT", {"default": 704, "min": 256, "max": 1080}),
			}
		}

	RETURN_TYPES = ("IMAGE", "STRING")
	FUNCTION = "generate"
	CATEGORY = "Cosmos"

	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# 🧠 Redirect legacy import path for Cosmos checkpoint compatibility
		import nodes.cosmos.cosmos_tokenizer.utils as actual_utils
		sys.modules["nodes.cosmos.utils"] = actual_utils

		# Paths to your model weights
		self.model_path = r"..."
		...

		# Instantiate and load weights
		from nodes.my_model_defs import CosmosModel, CosmosVAE, CosmosEncoder

		self.model = CosmosModel(
			max_img_h=704,
			max_img_w=1280,
			max_frames=24,
			in_channels=3,
			out_channels=3,
			patch_spatial=16,
			patch_temporal=2,
			operations=operations,
			pos_emb_cls=pos_emb_cls  # ✅ string, not class
		).to(self.device)



		self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))

		print(f"[Cosmos] All components loaded on {self.device}")



	def tensor_to_video_frames(self, tensor):
		frames = []
		for i in range(tensor.shape[0]):
			frame = tensor[i].cpu().clamp(0, 1) * 255
			frame = frame.permute(1, 2, 0).byte().numpy()
			frames.append(frame)
		return frames


	def generate(self, text_prompt, frame_count, width, height):
		with torch.no_grad():
			# Encode text
			text_embedding = self.encoder.encode(text_prompt)

			# Generate latent video
			latent_video = self.model.generate_video(
				text_embedding=text_embedding,
				num_frames=frame_count,
				width=width,
				height=height
			)

			# Decode with VAE
			video_tensor = self.vae.decode(latent_video)

			# Convert to preview frames
			frames = self.tensor_to_video_frames(video_tensor)

			# Save first frame for preview
			output_dir = os.path.join(r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\outputs", "cosmos_video")
			os.makedirs(output_dir, exist_ok=True)
			save_path = os.path.join(output_dir, "preview_frame.png")
			Image.fromarray(frames[0]).save(save_path)

			return (frames[0], save_path)

NODE_CLASS_MAPPINGS = {
	"CosmosTextToVideo": CosmosTextToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"CosmosTextToVideo": "Cosmos Text-to-Video"
}