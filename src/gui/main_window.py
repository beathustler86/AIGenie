import os
import os,time
import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime
from tkinter import filedialog
from PIL import Image, ImageTk
from diffusers.models import AutoencoderKL
import torch
from diffusers import StableDiffusionXLPipeline
from src.modules.refiner_module import refine_image
import json
import numpy as np
import torchvision.transforms as T
from src.nodes.cosmos_text_to_video import CosmosTextToVideo

import sys
#sys.path.append(r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\models\text_to_video\ComfyUI\custom_nodes\comfyui_cosmos")
#from nodes_cosmos import CosmosTextToVideo



from torchvision import transforms
from diffusers import (
	EulerDiscreteScheduler,
	DPMSolverMultistepScheduler,
	PNDMScheduler,
	DDIMScheduler,
	LCMScheduler
)
# === NEW Import for SD3.5 TensorRT ===
from src.nodes.cosmos_text_to_video import CosmosTextToVideo


# === TACTICAL PATHS ===
OUTPUT_DIR = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\outputs"
REFINED_DIR = os.path.join(OUTPUT_DIR, "images", "refined")
BIN_LOG = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\logs\bin_log.txt"
MODEL_ROOT = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_image"

def discover_models():
	models = {}
	for root, dirs, files in os.walk(MODEL_ROOT):
		for d in dirs:
			models[d] = os.path.join(root, d)
		for f in files:
			if f.endswith(".safetensors"):
				name = os.path.splitext(f)[0]
				models[name] = os.path.join(root, f)
	return models

# === MODEL CAPABILITIES MAP ===
MODEL_CAPABILITIES = {
	"SDXL 1.0": {"refiner": True},
	"SDXL 1.5": {"refiner": True},
	"SD3.5 TensorRT": {"refiner": False},
	"DreamShaper XL Turbo v2": {"refiner": True}  # ✅ Enabled for refinement
}

NODE_CLASS_MAPPINGS = {
	"CosmosTextToVideo": CosmosTextToVideo
}



# === TELEMETRY UTILITY GLOBAL SCOPE ===
def get_cuda_vram():
	try:
		props = torch.cuda.get_device_properties(0)
		total = props.total_memory // 1024**2
		used = torch.cuda.memory_allocated() // 1024**2
		return used, total
	except Exception as e:
		print(f"[Telemetry] CUDA VRAM fetch failed: {e}")
		return 0, 0

def get_model_metadata(path):
	return {
		"size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
		"last_modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))
	}

def log_model_switch(name, path):
	with open(r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\logs\model_switch.log", "a", encoding="utf-8") as log:
		log.write(f"[{datetime.now()}] Switched to {name} -> {path}\n")

	TELEMETRY_LOG = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\outputs\logs\telemetry_logs\telemetry.txt"
	with open(TELEMETRY_LOG, "a", encoding="utf-8") as log:
		log.write(f"[{datetime.now()}] Switched to {name} → {path}\n")


# =============== M A I N === W I N D O W === C L A S S  ==============
class MainWindow(tk.Frame):
	def __init__(self, root, sd35_sessions=None):
		super().__init__(root)  # ✅ Initialize tk.Frame
		self.root = root
		self.sd35_sessions = sd35_sessions or {}
		self.custom_save_path = tk.StringVar(value="")

		self.model_paths = {
			"SDXL 1.0": r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_image\sdxl-base-1.0",
			"SDXL 1.5": r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_image\sdxl-base-1.5",
			"SD3.5 TensorRT": r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_image\sd3_5_tensorrt",
			"DreamShaper XL Turbo v2": r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_image\dreamshaper-xl-v2-turbo",
			"Cosmos 7B Text2Video": r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_video\ComfyUI\models\checkpoints\Cosmos-1_0-Diffusion-7B-Text2World.safetensors"
		}



		self.selected_model = tk.StringVar(value="SDXL 1.0")
		
		self.sampler_map = {
			"Euler": EulerDiscreteScheduler,
			"DPM++": DPMSolverMultistepScheduler,
			"PNDM": PNDMScheduler,
			"DDIM": DDIMScheduler,
			"LCM": LCMScheduler
}

		self.ultrawide_presets = {
			"1280×544": (1280, 544),
			"2560×1080": (2560, 1080),
			"3440×1440": (3440, 1440)
		}

		self.standard_resolutions = ["512x512", "1024x1024", "2048x2048","4096X4096"]
		self.ultrawide_resolutions = ["1280x544", "2560x1080", "3440x1440"]
		
		
		self.sd35_sessions = sd35_sessions or {}
		self.custom_save_path = tk.StringVar(value="")

		self.setup_gui()
		self.root.after(100, self.wait_for_gui_ready)
		self.use_refiner = tk.BooleanVar(value=False)
		self.uploaded_image = None

	def build_bottom_panel(self):
		ttk.Label(self.root, textvariable=self.custom_save_path, font=("Segoe UI", 9)).pack(pady=(0, 5))

		bottom_panel = ttk.Frame(self.root)
		bottom_panel.pack(side="bottom", pady=10)

		self.video_button = ttk.Button(
			bottom_panel,
			text="🎞️ Generate Video",
			command=self.generate_video,
			state="disabled",
			style="Cockpit.TButton"
		)
		self.video_button.pack(side="left", padx=5)


	def display_image(self, image, overlay_text="🖼️ Image Generated"):
		image = image.resize((self.width_var.get(), self.height_var.get()))
		self.tk_image = ImageTk.PhotoImage(image)
		self.canvas.delete("all")
		self.canvas.create_image(
			self.width_var.get() // 2,
			self.height_var.get() // 2,
			image=self.tk_image,
			anchor="center"
		)
		self.canvas.create_text(
			self.width_var.get() // 2,
			self.height_var.get() - 30,
			text=overlay_text,
			fill="white",
			font=("Segoe UI", 12)
		)




	def run_refiner(self):
		if not hasattr(self, "generated_image") or self.generated_image is None:
			self.time_label.config(text="No image available for refinement.")
			return

		selected = self.selected_model.get()
		can_refine = MODEL_CAPABILITIES.get(selected, {}).get("refiner", False)
		if not can_refine:
			self.time_label.config(text="Refiner not supported for this model.")
			return

		prompt = self.last_generation_meta.get("prompt", "Refine pass")
		negative = self.last_generation_meta.get("negative", None)
		width, height = self.generated_image.size

		try:
			from src.modules.refiner_module import refine_image
			result = refine_image(
				base_image=self.generated_image,
				prompt=prompt,
				negative=negative,
				width=width,
				height=height,
				strength=0.3,
				save=True,
				device="cuda"
			)

			self.generated_image = result["image"]
			self.display_image(result["image"], overlay_text="🧪 Refined")
			self.save_image(result["image"])
			self.save_button.config(state="normal")
			self.refine_button.config(state="disabled", text="🧪 Refine (Complete)")
			self.after(2000, lambda: self.refine_button.config(state="normal", text="🧪 Refine Again"))
			self.time_label.config(text=f"Refined: {result['filename']}")
			self.update_telemetry_status(f"Refinement complete in {result['duration']}s → {result['filename']}")
			print(f"🧪 Refinement complete: {result['path']}")

		except Exception as e:
			self.time_label.config(text=f"Refinement failed: {e}")
			print(f"❌ Refiner error: {e}")

	def cosmos_task(self, prompt, frame_count, width, height):
		try:
			self.telemetry_label.config(text="🎬 Cosmos generating...")
			self.cosmos_node.threaded_generate(prompt, frame_count, width, height)
			self.telemetry_label.config(text="✅ Cosmos generation dispatched.")
		except Exception as e:
			self.telemetry_label.config(text=f"[Error] Cosmos generation failed: {e}")
			print(f"[Error] Cosmos generation failed: {e}")



	def upscale_image(self, model_choice):
		self.refine_button.config(text="🧪 Refining...", state="disabled")
		threading.Thread(target=self.run_refiner, daemon=True).start()

	def threaded_upscale(self):
		self.upscale_button.config(text="🔼 Upscaling...", state="disabled")
		selected_model = self.upscale_model_var.get()
		threading.Thread(target=lambda: self.upscale_image(selected_model), daemon=True).start()

	def threaded_refine(self):
		self.refine_button.config(text="🧪 Refining...", state="disabled")
		threading.Thread(target=self.run_refiner, daemon=True).start()

	def generate_cosmos_threaded(self, prompt, frame_count, width, height):
		threading.Thread(
			target=lambda: self.cosmos_task(prompt, frame_count, width, height),
			daemon=True
		).start()

	def on_model_selected(self, selected_model):
		self.switch_model_threaded(selected_model)
		self.update_model_selection(selected_model)


		
	def update_resolution_menu(self):
		menu = self.resolution_menu["menu"]
		menu.delete(0, "end")

		options = self.ultrawide_resolutions if self.ultrawide_var.get() else self.standard_resolutions
		default = options[0]
		self.resolution_var.set(default)

		for res in options:
			menu.add_command(label=res, command=lambda value=res: self.resolution_var.set(value))


	def wait_for_gui_ready(self):
		if hasattr(self, "time_label"):
			threading.Thread(target=self.load_model, args=(self.selected_model.get(),), daemon=True).start()
		else:
			self.root.after(100, self.wait_for_gui_ready)

	def load_model(self, selection):
		if not hasattr(self, "time_label"):
			print(f"⚠️ time_label not initialized — skipping GUI update")
			return
	


		path = self.model_paths.get(selection)
		if selection == "SD3.5 TensorRT":
			try:
				self.time_label.config(text=f"Loading {selection}...")
				print(f"🔁 Loading {selection} from {path}")

				from src.models.sd3_5_tensorrt.sd35_loader import load_sd35_main, initialize_sd35_modules

				session = load_sd35_main()
				if session is None:
					print("🔁 SD3.5 fallback to modular pipeline")
					self.sd35_sessions = initialize_sd35_modules()
					session = self.sd35_sessions.get("transformer_fp8")
				else:
					self.sd35_sessions["session"] = session

				self.time_label.config(text=f"{selection} session ready.")
				self.update_canvas_status()
			except Exception as e:
				self.time_label.config(text=f"Model load failed: {e}")
				print(f"❌ SD3.5 load error: {e}")
		elif selection == "DreamShaper XL Turbo v2":
			try:
				self.time_label.config(text=f"Loading {selection}...")
				print(f"🔁 Loading {selection} from {path}")

				del self.pipe
				torch.cuda.empty_cache()

				self.pipe = StableDiffusionXLPipeline.from_pretrained(
					path,
					torch_dtype=torch.float16,
					variant="fp16"
				).to("cuda")

				self.pipe.vae = AutoencoderKL.from_pretrained(
					"madebyollin/sdxl-vae-fp16-fix",
					torch_dtype=torch.float16
				).to("cuda")

				self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

				self.time_label.config(text=f"{selection} loaded.")
				self.update_canvas_status()
			except Exception as e:
				self.time_label.config(text=f"Model load failed: {e}")
				print(f"❌ DreamShaper load error: {e}")
		elif path:
			try:
				self.time_label.config(text=f"Loading {selection}...")
				print(f"🔁 Loading {selection} from {path}")

				if selection == "Cosmos 7B Text2Video":
					from safetensors.torch import load_file
					from src.nodes.cosmos_text_to_video import CosmosTextToVideo

					model = CosmosTextToVideo()
					state_dict = load_file(path)
					model.model.load_state_dict(state_dict)

					self.cosmos_model = model  # Store for later use
					self.time_label.config(text=f"{selection} loaded and telemetry confirmed.")
					print("✅ Cosmos 7B weights injected.")
					self.update_canvas_status()
				else:
					self.pipe = StableDiffusionXLPipeline.from_pretrained(
						path,
						torch_dtype=torch.float16,
						variant="fp16"
					).to("cuda")

					self.pipe.vae = AutoencoderKL.from_pretrained(
						"madebyollin/sdxl-vae-fp16-fix",
						torch_dtype=torch.float16
					).to("cuda")

					self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)

					self.time_label.config(text=f"{selection} loaded.")
					self.update_canvas_status()
			except Exception as e:
				self.time_label.config(text=f"Model load failed: {e}")
				print(f"❌ Model load error: {e}")


	def update_canvas_status(self):
		self.canvas.delete("status_label")
		self.canvas.delete("model_label")

		refiner_text = "Refined" if self.use_refiner.get() else "Base"
		self.canvas.create_text(
			10, 10,
			anchor="nw",
			text=refiner_text,
			fill="white",
			font=("Segoe UI", 10, "bold"),
			tags="status_label"
		)

		model_text = f"Model: {self.selected_model.get()}"
		self.canvas.create_text(
			10, 30,
			anchor="nw",
			text=model_text,
			fill="lightgray",
			font=("Segoe UI", 9),
			tags="model_label"
		)

		self.sampler_map = {
			"Euler": EulerDiscreteScheduler,
			"DPM++": DPMSolverMultistepScheduler,
			"PNDM": PNDMScheduler,
			"DDIM": DDIMScheduler,
			"LCM": LCMScheduler
		}

	def update_telemetry_status(self, message="Telemetry online."):
		print(f"[Telemetry] {message}")

	# === GUI SETUP FULL (panels, canvas, sliders etc.) ===

	def setup_gui(self):
		main_frame = tk.Frame(self.root)
		main_frame.pack(expand=True, fill="both")

		left_panel = tk.Frame(main_frame)
		left_panel.pack(side="left", fill="y", padx=20, pady=20)

		center_panel = tk.Frame(main_frame)
		center_panel.pack(side="left", fill="both", expand=False)
		
		padded_wrapper = tk.Frame(center_panel, bg="#000000", padx=5, pady=5)
		padded_wrapper.pack(expand=True)

		bottom_panel = tk.Frame(main_frame)
		bottom_panel.pack(side="bottom", fill="x", padx=10, pady=10)



		# === REFINE BUTTON ===
		self.refine_button = ttk.Button(
			bottom_panel,
			text="🧪 Refine",
			command=self.threaded_refine,
			state="disabled",
			style="Cockpit.TButton"
		)

		self.refine_button.pack(side="left", padx=5)

		self.width_var = tk.IntVar(value=1920)
		self.height_var = tk.IntVar(value=1080)

		self.canvas = tk.Canvas(
			padded_wrapper,
			width=self.width_var.get(),
			height=self.height_var.get(),
			bg="#1e1e1e",
			highlightbackground="#000000",  # Optional: black border
			highlightthickness=1			# Optional: border thickness
		)
		self.canvas.pack(anchor="center")


		self.canvas.create_text(
			self.width_var.get() // 2,
			self.height_var.get() // 2,
			text="🧠 Cockpit Ready",
			fill="white",
			font=("Segoe UI", 16)
		)






		# === LEFT PANEL ===

		# === ENABLE REFINER TOGGLE ===
		self.use_refiner = tk.BooleanVar(value=True)
		self.refiner_toggle = ttk.Checkbutton(left_panel, text="Enable Refiner", variable=self.use_refiner)
		self.refiner_toggle.pack(anchor="w", pady=(10, 0))
		# === ULTRAWIDE TOGGLE ===
		self.resolution_var = tk.StringVar(value="1024x1024") 
		ttk.Label(left_panel, text="Render Size").pack(anchor="w", pady=(10, 0))
		self.resolution_menu = ttk.OptionMenu(left_panel, self.resolution_var, "512x512", *self.standard_resolutions)
		
		self.resolution_menu.pack(anchor="w", fill="x")
		self.ultrawide_var = tk.BooleanVar(value=False)
		ttk.Checkbutton(
			left_panel,
			text="Ultrawide (21:9)",
			variable=self.ultrawide_var,
			command=self.update_resolution_menu
		).pack(anchor="w", pady=(5, 0))

		try:
			from src.modules.refiner_module import REFINER_AVAILABLE
			status_text = "🧠 Refiner: Active" if REFINER_AVAILABLE else "🧪 Refiner: Fallback"
			status_color = "green" if REFINER_AVAILABLE else "orange"
			ttk.Label(left_panel, text=status_text, foreground=status_color).pack(anchor="w", pady=(5, 0))
		except Exception:
			ttk.Label(left_panel, text="🧪 Refiner: Unknown", foreground="gray").pack(anchor="w", pady=(5, 0))

		# === SAMPLER VARIABLES ===
		self.sampler_var = tk.StringVar(value="Euler")
		ttk.OptionMenu(
			left_panel,
			self.sampler_var,
			"Euler",
			"Euler", "Euler a", "DPM++ 2M", "DPM++ SDE", "DDIM", "Heun", "LMS",
			command=self.update_sampler_info
		).pack(anchor="w", fill="x")

		ttk.Label(left_panel, text="Refiner Strength").pack(anchor="w", pady=(10, 0))
		self.strength_var = tk.DoubleVar(value=0.3)
		self.strength_label = ttk.Label(left_panel, text=f"Strength: {self.strength_var.get():.2f}")
		self.strength_label.pack(anchor="w")
		tk.Scale(left_panel, from_=0.1, to=1.0, resolution=0.05, variable=self.strength_var, orient="horizontal").pack(anchor="w", fill="x")
		self.strength_var.trace_add("write", lambda *args: self.strength_label.config(
			text=f"Strength: {self.strength_var.get():.2f}"
		))

		# === SAMPLER INFO TOGGLE ===
		self.show_sampler_info = tk.BooleanVar(value=False)
		ttk.Checkbutton(
			left_panel,
			text="Show Sampler Info",
			variable=self.show_sampler_info,
			command=self.toggle_sampler_info
		).pack(anchor="w", pady=(5, 0))

		# === SAMPLER INFO PANEL ===
		self.sampler_info_frame = ttk.Frame(left_panel)
		self.sampler_info_label = ttk.Label(
			self.sampler_info_frame,
			text="Select a sampler to view its traits.",
			wraplength=300,
			justify="left"
		)
		self.sampler_info_label.pack()
		self.sampler_info_frame.pack(anchor="w", fill="x", pady=(5, 0))
		self.sampler_info_frame.pack_forget()

		# === MODEL SELECTION ===
		ttk.Label(left_panel, text="🧠 Model").pack(anchor="w", pady=(10, 0))
		model_dropdown = ttk.OptionMenu(
			left_panel,
			self.selected_model,
			self.selected_model.get(),
			*self.model_paths.keys(),
			command=self.switch_model_threaded  # ✅ Threaded callback
		)
		model_dropdown.pack(anchor="w")
		self.model_dropdown = model_dropdown 
		self.telemetry_label = ttk.Label(left_panel, text="🧠 Model telemetry will appear here")
		self.telemetry_label.pack(anchor="w", pady=(5, 0))

		# === UPLOAD IMAGE BUTTON ===
		ttk.Label(left_panel, text="📤 Upload Image", font=("Segoe UI", 12)).pack(anchor="w", pady=(20, 0))
		ttk.Button(left_panel, text="Upload Image", command=self.upload_image).pack(anchor="w", pady=5)

		# === PROMPT FIELDS ===
		ttk.Label(left_panel, text="📝 Prompt").pack(anchor="w", pady=(20, 0))
		self.prompt_entry = tk.Text(left_panel, height=4, width=40)
		self.prompt_entry.pack(anchor="w")

		ttk.Label(left_panel, text="🚫 Negative Prompt").pack(anchor="w", pady=(20, 0))
		self.negative_prompt_entry = tk.Text(left_panel, height=4, width=40)
		self.negative_prompt_entry.pack(anchor="w")

		# === GENERATION SLIDERS ===
		ttk.Label(left_panel, text="CFG Scale").pack(anchor="w", pady=(10, 0))
		self.cfg_scale_var = tk.DoubleVar(value=7.5)
		self.cfg_scale_label = ttk.Label(left_panel, text=f"CFG Scale: {self.cfg_scale_var.get():.1f}")
		self.cfg_scale_label.pack(anchor="w")
		ttk.Scale(left_panel, from_=1.0, to=20.0, variable=self.cfg_scale_var, orient="horizontal").pack(anchor="w", fill="x")
		self.cfg_scale_var.trace_add("write", lambda *args: self.cfg_scale_label.config(
			text=f"CFG Scale: {self.cfg_scale_var.get():.1f}"
		))

		self.steps_var = tk.IntVar(value=50)
		self.steps_label = ttk.Label(left_panel, text=f"Steps: {self.steps_var.get()}")
		self.steps_label.pack(anchor="w", pady=(10, 0))
		ttk.Scale(left_panel, from_=10, to=150, variable=self.steps_var, orient="horizontal").pack(anchor="w", fill="x")
		self.steps_var.trace_add("write", lambda *args: self.steps_label.config(
			text=f"Steps: {self.steps_var.get()}"
		))

		ttk.Label(left_panel, text="Seed").pack(anchor="w", pady=(10, 0))
		self.seed_var = tk.IntVar(value=0)
		self.seed_label = ttk.Label(left_panel, text=f"Seed: {self.seed_var.get()}")
		self.seed_label.pack(anchor="w")
		ttk.Entry(left_panel, textvariable=self.seed_var).pack(anchor="w", fill="x")
		self.seed_var.trace_add("write", lambda *args: self.seed_label.config(
			text=f"Seed: {self.seed_var.get()}"
		))

		self.batch_count_var = tk.IntVar(value=1)
		self.batch_label = ttk.Label(left_panel, text=f"Batch Count: {self.batch_count_var.get()}")
		self.batch_label.pack(anchor="w", pady=(10, 0))
		ttk.Scale(left_panel, from_=1, to=8, variable=self.batch_count_var, orient="horizontal").pack(anchor="w", fill="x")
		self.batch_count_var.trace_add("write", lambda *args: self.batch_label.config(
			text=f"Batch Count: {self.batch_count_var.get()}"
		))

# === BOTTOM PANEL ===
		ttk.Label(self.root, textvariable=self.custom_save_path, font=("Segoe UI", 9)).pack(pady=(0, 5))
		bottom_panel = ttk.Frame(self.root)
		bottom_panel.pack(side="bottom", pady=10)

# === UPSCALE MODEL SELECTOR ===
		self.upscale_model_var = tk.StringVar(value="Remacri")
		self.upscale_model_dropdown = ttk.Combobox(
			bottom_panel,
			textvariable=self.upscale_model_var,
			values=["UltraSharp", "Remacri", "Anime6B"],
			state="readonly",
			width=10
		)
		self.upscale_model_dropdown.pack(side="left", padx=5)
	def update_model_selection(self, selected_model):
		is_comfy = "comfy" in selected_model.lower()
		self.video_button.config(state="normal" if is_comfy else "disabled")
		self.update_telemetry_status(f"🧠 Selected model: {selected_model} {'(ComfyUI detected)' if is_comfy else ''}")

# === GENERATE BUTTON ===
		ttk.Button(bottom_panel, text="🚀 Generate", command=self.threaded_generate).pack(side="left", padx=5)
		self.refine_button = ttk.Button(
			bottom_panel,
			text="🧪 Refine",
			command=self.threaded_refine,
			state="disabled",
			style="Cockpit.TButton"
		)
		self.refine_button.pack(side="left", padx=5)
		
		

# === GENERATE VIDEO BUTTON ===
		ttk.Button(bottom_panel, text="🚀 Generate Video", command=self.threaded_generate).pack(side="left", padx=5)
		self.video_button = ttk.Button(
			bottom_panel,
			text="🎞️ Generate Video",
			command=self.generate_video,
			state="disabled",
			style="Cockpit.TButton"
		)
		self.video_button.pack(side="left", padx=5)




# === SAVE BUTTON ===
		self.save_button = ttk.Button(
			bottom_panel,
			text="💾 Save Image",
			command=self.save_image,
			state="disabled",  # Optional: disable until image is ready
			style="Cockpit.TButton"
		)
		self.save_button.pack(side="left", padx=5)
		

# === UPSCALE IMAGE BUTTON ===
		self.upscale_button = ttk.Button(
			bottom_panel,
			text="🔼 Upscale Image",
			command=self.threaded_upscale,
			state="normal",
			style="Cockpit.TButton"
		)
		self.upscale_button.pack(side="left", padx=5)


# === BIN IMAGE BUTTON ===
		ttk.Button(bottom_panel, text="🗑️ Bin Image", command=self.bin_image).pack(side="left", padx=5)
		

# === CHOOSE SAVE LOCATION BUTTON ===
		def choose_save_location():
			selected_dir = filedialog.askdirectory()
			if selected_dir:
				self.custom_save_path.set(selected_dir)
		ttk.Button(bottom_panel, text="📁 Choose Save Location", command=choose_save_location).pack(side="left", padx=5)

		self.time_label = ttk.Label(bottom_panel, text="Status: Waiting for image")
		self.time_label.pack(side="left", padx=20)
		ttk.Button(bottom_panel, text="📊 View Telemetry", command=self.view_telemetry).pack(side="left", padx=5)

		self.progress_var = tk.DoubleVar(value=0)
		self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100, mode="determinate")
		self.progress_bar.pack(side="bottom", fill="x", padx=20, pady=5)

		logo_path = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\src\data\poweredbynvidia_optimized.png"
		logo_img = Image.open(logo_path)
		logo_preview = logo_img.resize((120, 40))
		self.logo_tk = ImageTk.PhotoImage(logo_preview)
		self.logo_label = tk.Label(self.root, image=self.logo_tk, bg="gray20")
		self.logo_label.place(relx=0.0, rely=1.0, anchor="sw", x=20, y=-10)

	def toggle_sampler_info(self):
		if self.show_sampler_info.get():
			self.sampler_info_frame.pack(anchor="w", fill="x", pady=(5, 0))
			self.update_sampler_info(self.sampler_var.get())
		else:
			self.sampler_info_frame.pack_forget()

	def update_sampler_info(self, sampler_name):
		info_map = {
			"Euler": "Fast and basic. Crisp edges, good for simple scenes.",
			"Euler a": "Ancestral variant. More creative, sometimes chaotic.",
			"DPM++ 2M": "Smooth gradients, photorealistic fidelity.",
			"DPM++ SDE": "High detail, great for realism.",
			"DDIM": "Fast, less variation, good for batch consistency.",
			"Heun": "Balanced quality and speed.",
			"LMS": "Older, stylized outputs, sometimes useful."
		}
		self.sampler_info_label.config(text=info_map.get(sampler_name, "No info available."))

	def switch_model_threaded(self, selection):
		def task():
			try:
				self.telemetry_label.config(text=f"🔁 Loading {selection}...")
				start = time.time()
				self.load_model(selection)
				print(f"⏱️ load_model() took {time.time() - start:.2f}s")


				can_refine = MODEL_CAPABILITIES.get(selection, {}).get("refiner", False)
				self.use_refiner.set(can_refine)
				self.refiner_toggle.config(state="normal" if can_refine else "disabled")

				model_path = self.model_paths.get(selection)
				if model_path:
					metadata = get_model_metadata(model_path)
					self.telemetry_label.config(
						text=f"🧠 {selection}\n📦 {metadata['size_mb']} MB | 🕒 Modified: {metadata['last_modified']}"
					)
					log_model_switch(selection, model_path)

				if hasattr(self, "model_presets"):
					preset = self.model_presets.get(selection)
					if preset:
						self.cfg_scale_var.set(preset["cfg"])
						self.steps_var.set(preset["steps"])
						self.sampler_var.set(preset["sampler"])

				if selection == "Cosmos 7B Text2Video":
					try:
						start = time.time()
						self.cosmos_node = CosmosTextToVideo()
						print(f"⏱️ Cosmos init took {time.time() - start:.2f}s")

						can_refine = False
						self.time_label.config(text=f"{selection} module loaded.")
						print(f"🎬 Cosmos module ready.")
					except Exception as e:
						self.telemetry_label.config(text=f"[Error] Cosmos load failed: {e}")
						print(f"[Error] Cosmos load failed: {e}")


				self.time_label.config(text=f"{selection} selected — Refiner {'enabled' if can_refine else 'disabled'}")
				print(f"🧠 Model switched to: {selection} | Refiner {'enabled' if can_refine else 'disabled'}")

			except Exception as e:
				self.telemetry_label.config(text=f"[Error] Model swap failed: {e}")
				print(f"[Error] Model swap failed: {e}")

		threading.Thread(target=task, daemon=True).start()





	# === UPLOAD IMAGE ===
	def upload_image(self):
		file_path = filedialog.askopenfilename(
			title="Select Image",
			filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
		)
		if file_path:
			try:
				img = Image.open(file_path)
				self.uploaded_image = img
				preview = img.resize((256, 256))
				img_tk = ImageTk.PhotoImage(preview)
				self.update_canvas(img_tk, img)
				self.time_label.config(text=f"Uploaded: {os.path.basename(file_path)}")
				print(f"📤 Image uploaded: {file_path}")
			except Exception as e:
				self.time_label.config(text=f"Upload failed: {e}")
				print(f"❌ Upload error: {e}")

	# === THREADED GENERATE ===
	def threaded_generate(self):
		thread = threading.Thread(target=self.generate_image)
		thread.start()

# === GENERATE IMAGE ===
	def generate_image(self):
		sampler_name = self.sampler_var.get()
		scheduler_cls = self.sampler_map.get(sampler_name, EulerDiscreteScheduler)

		selected = self.selected_model.get()
		model_path = self.model_paths.get(selected)

		if not hasattr(self, "pipe") or self.pipe is None:
			print("❌ Pipeline not initialized. Aborting generation.")
			return

		self.refine_button.config(state="disabled")
		self.refine_button.config(text="🧪 Refine")

		self.pipe.scheduler = scheduler_cls.from_config(self.pipe.scheduler.config)

		self.canvas.delete("all")
		self.canvas.image = None

		if hasattr(self, "progress_var"):
			self.progress_var.set(0)
			self.progress_bar.start(10)

		self.update_telemetry_status("Image generation started...")

		prompt = self.prompt_entry.get("1.0", "end").strip()
		negative = self.negative_prompt_entry.get("1.0", "end").strip()
		can_refine = MODEL_CAPABILITIES.get(selected, {}).get("refiner", False)
		use_refiner = self.use_refiner.get() and can_refine
		start_time = datetime.now()

# === GENERATE VIDEO ====
	def generate_video(self):
		selected = self.selected_model.get()
		model_path = self.model_paths.get(selected)

		if not hasattr(self, "cosmos_node") or self.cosmos_node is None:
			print("❌ Cosmos node not initialized.")
			return

		prompt = self.prompt_entry.get("1.0", "end").strip()
		frame_count = 16
		width, height = 512, 512

		self.generated_frames = []
		self.update_telemetry_status("🎞️ Cosmos video generation started...")

		def on_frame(frame):
			self.generated_frames.append(frame)

		self.cosmos_node.threaded_generate(prompt, frame_count, width, height, callback=on_frame)

		# Optional: trigger preview after short delay
		self.root.after(2000, lambda: self.play_video_preview())


		# === RESOLUTION FROM DROPDOWN ===
		selected_res = self.resolution_var.get()  # e.g. "1920x1080"
		try:
			width, height = map(int, selected_res.split("x"))
			self.width_var.set(width)
			self.height_var.set(height)
		except ValueError:
			print(f"⚠️ Invalid resolution format: {selected_res}")
			width = int(self.width_var.get())
			height = int(self.height_var.get())

		print(f"[Resolution] Width: {width} | Height: {height}")

		with torch.no_grad():
			base_result = self.pipe(
				prompt,
				negative_prompt=negative,
				width=width,
				height=height,
				num_inference_steps=self.steps_var.get(),
				guidance_scale=self.cfg_scale_var.get(),
				generator=torch.manual_seed(self.seed_var.get()) if self.seed_var.get() > 0 else None,
				sampler=sampler_name
			)

		print(f"🧩 Generated image size: {base_result.images[0].size}")

		self.generated_image = base_result.images[0]
		self.display_image(self.generated_image, overlay_text="🖼️ Base Image")
		self.save_image(self.generated_image)

		self.last_generation_meta = {
			"prompt": prompt,
			"negative": negative,
			"model": selected,
			"time": start_time.isoformat()
		}

		self.refine_button.config(state="normal")
		self.refine_button.config(text="🧪 Refine (Ready)")
		self.update_telemetry_status("Image generated. Ready for optional refinement.")  # ✅ Insert here


		end_time = datetime.now()
		duration = (end_time - start_time).total_seconds()

		if hasattr(self, "progress_bar"):
			self.progress_bar.stop()
		if hasattr(self, "progress_var"):
			self.progress_var.set(100)

		self.update_telemetry_status(f"Generation complete in {duration:.2f}s")

		log_path = os.path.join(OUTPUT_DIR, "logs", "generation_log.txt")
		os.makedirs(os.path.dirname(log_path), exist_ok=True)
		with open(log_path, "a", encoding="utf-8") as log:
			log.write(f"{datetime.now()}: Resolution={width}x{height} | Prompt='{prompt}' | Refiner={'ON' if use_refiner else 'OFF'} | Duration={duration:.2f}s\n")

		self.time_label.config(text=f"✅ {selected} model operational.")


	# === UPDATE CANVAS ===
	def update_canvas(self, img_tk, final_image):
		canvas_width = self.canvas.winfo_width()
		canvas_height = self.canvas.winfo_height()
		resized = final_image.resize((canvas_width, canvas_height), Image.LANCZOS)
		img_tk = ImageTk.PhotoImage(resized)
		self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
		self.canvas.image = img_tk
		self.uploaded_image = final_image
		self.canvas.create_text(
			10, 10,
			anchor="nw",
			text="Refined" if self.use_refiner.get() else "Base",
			fill="white",
			font=("Segoe UI", 10, "bold")
		)

		save_dir = "output/refined" if self.use_refiner.get() else "output/base"
		os.makedirs(save_dir, exist_ok=True)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = f"{save_dir}/render_{timestamp}.png"
		self.uploaded_image.save(filename)
		self.update_telemetry_status(f"🖼️ Saved full-res render to {filename}")

# === SAVE IMAGE ===
	def save_image(self, image=None):
		image_to_save = image or self.generated_image or self.uploaded_image
		if image_to_save:
			save_dir = self.custom_save_path.get().strip()
			if not save_dir:
				save_dir = os.path.join(OUTPUT_DIR, "images")
			os.makedirs(save_dir, exist_ok=True)

			width, height = image_to_save.size
			model_tag = ""
			if hasattr(self, "upscale_model_var"):
				model_tag = f"_{self.upscale_model_var.get()}"

			filename = f"image{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{width}x{height}.png"
			save_path = os.path.join(save_dir, filename)

			image_to_save.save(save_path)
			self.last_saved_path = save_path

			print(f"🖼️ Saved image to: {save_path}")
			print(f"📐 Image size: {image_to_save.size}")
			self.time_label.config(text=f"Saved: {filename}")
			self.update_telemetry_status(f"🖼️ Saved full-res render to {save_path}")
		else:
			print("⚠️ No image to save.")
			self.time_label.config(text="Save failed: No image loaded.")


# === UPSCALE IMAGE ===
	def upscale_image(self, model_choice):
		import torch, time

		if not hasattr(self, "generated_image") or self.generated_image is None:
			self.time_label.config(text="No image available to upscale.")
			return

		try:
			from src.modules.upscale_module import upscale_image_pass

			self.time_label.config(text=f"🔼 Upscaling with {model_choice}...")
			start_time = time.time()
			device = "cuda" if torch.cuda.is_available() else "cpu"
			print(f"[COCKPIT] Launching upscale → Model: {model_choice}")

			result = upscale_image_pass(
				image=self.generated_image,
				model_name=model_choice,
				scale=4,
				device=device,
				save=False
			)

			print(f"[COCKPIT] Upscale result received → Keys: {list(result.keys())}")

			if "config" in result:
				print(f"[COCKPIT] Model config → {result['config']}")

			w, h = result["image"].size
			duration = round(time.time() - start_time, 2)
			print(f"[COCKPIT] Final image size → {w}×{h}")

			self.generated_image = result["image"]
			self.display_image(result["image"], overlay_text=f"🔼 {model_choice}")
			self.save_image(result["image"])
			self.time_label.config(text=f"Upscaled in {duration}s → {result['filename']}")
			self.update_telemetry_status(f"✅ Upscale complete → {result['filename']}")
			self.log_event(f"Upscale complete → {result['filename']} | {w}×{h} | Model: {model_choice} | Device: {device} | Duration: {duration}s")
			print(f"[COCKPIT] Image saved → Path: {result['path']}")

		except Exception as e:
			self.time_label.config(text=f"Upscale failed: {e}")
			print(f"[ERROR] Upscale failed → {e}")







	# === BIN IMAGE ===
	def bin_image(self):
		self.canvas.delete("all")
		self.canvas.image = None
		self.uploaded_image = None
		self.last_saved_path = None
		self.time_label.config(text="Image binned.")
		with open(BIN_LOG, "a") as f:
			f.write(f"{datetime.now()}: Image binned\n")
		print("🗑️ Image binned and logged")

# === VIEW TELEMETRY ===
	def view_telemetry(self):
		telemetry_window = tk.Toplevel(self.root)
		telemetry_window.title("Telemetry Dashboard")
		telemetry_window.geometry("300x200")
		telemetry_window.attributes("-topmost", True)

		labels = {}
		for key in ["GPU", "CUDA Visible VRAM", "GPU Memory Used", "Max GPU Memory", "Batch Count", "Refiner", "Image Dimensions", "Model Config"]:
			labels[key] = ttk.Label(telemetry_window, text=f"{key}: ...")
			labels[key].pack(anchor="w", pady=5)

		def update_stats():
			mem_used = torch.cuda.memory_allocated() // 1024**2
			mem_max = torch.cuda.max_memory_allocated() // 1024**2
			batch = self.batch_count_var.get()
			refiner_status = "Enabled" if self.use_refiner.get() else "Disabled"
			used, total = get_cuda_vram()
			gpu_name = torch.cuda.get_device_name(0)

			stats = {
				"GPU": gpu_name,
				"CUDA Visible VRAM": f"{total} MB",
				"GPU Memory Used": f"{used} MB",
				"Max GPU Memory": f"{mem_max} MB",
				"Batch Count": batch,
				"Refiner": refiner_status,
				"Image Dimensions": f"{self.generated_image.size[0]}×{self.generated_image.size[1]}" if hasattr(self, "generated_image") and self.generated_image else "N/A",
				"Model Config": self.last_model_config if hasattr(self, "last_model_config") else "N/A"
			}
			for key, value in stats.items():
				labels[key].config(text=f"{key}: {value}")
			self.log_event(f"Telemetry update → {stats}")
			telemetry_window.after(1000, update_stats)

		update_stats()



