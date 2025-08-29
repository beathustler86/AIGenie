# ============================
# 🧠 refiner_module.py — SDXL Refiner Subsystem
# ============================

import torch
from PIL import Image
import os
import json
from datetime import datetime
import torchvision.transforms as T
import traceback




# === CONFIG ===
REFINER_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/sdxl-refiner-1.0"
SAVE_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/output/refined"
PROMPT = "cockpit-grade GUI with tactical overlays"

# === Attempt Import ===
try:
	from diffusers import StableDiffusionXLImg2ImgPipeline
	REFINER_IMPORTABLE = True
except ImportError:
	REFINER_IMPORTABLE = False

REFINER_AVAILABLE = os.path.isdir(REFINER_PATH) and REFINER_IMPORTABLE

def log_event(event):
	with open("telemetry_log.jsonl", "a") as f:
		if isinstance(event, dict):
			f.write(json.dumps(event) + "\n")
		else:
			f.write(json.dumps({"message": event}) + "\n")

def log_telemetry(**kwargs):
	log_event({**kwargs, "timestamp": datetime.now().isoformat()})

def log_memory(device):
	if device == "cuda":
		mem = torch.cuda.memory_allocated() / 1024**2
		log_event(f"[Refiner] CUDA memory used: {mem:.2f} MB")

# === INIT REFINE PIPELINE ===


def load_refiner(device="cuda"):
	if not REFINER_AVAILABLE:
		log_event({
			"event": "RefinerUnavailable",
			"path": REFINER_PATH,
			"device": device,
			"timestamp": datetime.now().isoformat()
		})
		raise RuntimeError("Refiner pipeline not available.")

	print("[Refiner] Loading img2img model...")
	pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
		REFINER_PATH,
		torch_dtype=torch.float16,
		variant="fp16",
		use_safetensors=True
	).to(device)

	log_model_config(pipe)
	print("[Refiner] Ready.")
	return pipe


def log_model_config(pipe):
	config_path = os.path.join("output", "refined", "refiner_config.json")
	os.makedirs(os.path.dirname(config_path), exist_ok=True)
	try:
		with open(config_path, "w") as f:
			json.dump(dict(pipe.config), f, indent=2)
		log_event(f"[Refiner] Config saved to {config_path}")
		print(f"[Refiner] ✅ Config written to: {config_path}")
	except Exception as e:
		log_event({
			"event": "RefinerConfigWriteError",
			"error": str(e),
			"path": config_path,
			"timestamp": datetime.now().isoformat()
		})
		print(f"[Refiner] ❌ Failed to write config: {e}")


def get_refiner_status():
	return {
		"available": REFINER_AVAILABLE,
		"device": "cuda" if torch.cuda.is_available() else "cpu",
		"path": REFINER_PATH,
		"importable": REFINER_IMPORTABLE,
		"timestamp": datetime.now().isoformat()
	}

def refine_image(base_image, prompt=PROMPT, negative=None, width=None, height=None, strength=0.3, save=True, save_path=SAVE_PATH, filename="refined_gui.png", device="cuda"):
	if device == "cuda" and not torch.cuda.is_available():
		print("[Refiner] CUDA not available — switching to CPU.")
		device = "cpu"

	if filename == "refined_gui.png":
		filename = f"refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

	# === Fallback Mode ===
	if not REFINER_AVAILABLE:
		print("⚠️ Refiner pipeline unavailable — using fallback.")
		duration = 0.0
		output_file = None

		if save:
			os.makedirs(save_path, exist_ok=True)
			output_file = os.path.join(save_path, filename)
			base_image.save(output_file)
			print(f"[Fallback] Saved to {output_file}")

		log_telemetry(
			event="RefinerFallback",
			duration=duration,
			prompt=prompt,
			device="stub",
			filename=filename,
			output_size=base_image.size,
			path=output_file,
			saved=save,
			variant="none",
			mode=base_image.mode,
			resolution=base_image.size
		)

		return {
			"image": base_image,
			"path": output_file,
			"prompt": prompt,
			"device": "stub",
			"duration": duration,
			"filename": filename
		}

	# === Refiner Mode ===
	refiner = load_refiner(device)
	log_model_config(refiner)
	log_event({
		"event": "RefinerLoaded",
		"path": REFINER_PATH,
		"device": device,
		"timestamp": datetime.now().isoformat()
	})
	print("[Refiner] Refining image...")

	import time
	start = time.time()

	if base_image.mode != "RGB":
		base_image = base_image.convert("RGB")

	# === Resize for Tactical Speed ===
	if width and height:
		target_size = (width, height)
		base_image = base_image.resize(target_size, Image.LANCZOS)
		print(f"[Refiner] Resized to GUI-specified resolution: {target_size}")
	else:
		target_size = (base_image.width, base_image.height)
		print(f"[Refiner] Using original image resolution: {target_size}")

	log_event({
		"event": "RefinerResize",
		"original_size": (target_size[0], target_size[1]),
		"resized_to": base_image.size,
		"timestamp": datetime.now().isoformat()
	})

	try:
		refined = refiner(
			prompt=prompt,
			image=base_image,
			strength=strength,
			num_inference_steps=20
		).images[0]

	except Exception as e:
		log_event({
			"event": "RefinerError",
			"error": str(e),
			"timestamp": datetime.now().isoformat(),
			"prompt": prompt,
			"device": device,
			"filename": filename,
			"log_trace": traceback.format_exc()
		})

		raise

	log_event(f"[Refiner] Output size: {refined.size}")

	duration = round(time.time() - start, 2)
	log_memory(device)

	output_file = None
	if save:
		os.makedirs(save_path, exist_ok=True)
		output_file = os.path.join(save_path, filename)
		refined.save(output_file)
		print(f"[Refiner] Saved to {output_file}")

	log_telemetry(
		event="Refiner",
		duration=duration,
		prompt=prompt,
		device=device,
		filename=filename,
		output_size=refined.size,
		path=output_file,
		saved=save,
		variant="fp16",
		mode=refined.mode,
		resolution=refined.size
	)

	return {
		"image": refined,
		"path": output_file,
		"prompt": prompt,
		"device": device,
		"duration": duration,
		"filename": filename
	}

