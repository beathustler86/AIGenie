# ============================
# 🧠 preflight_check.py — SDXL Cockpit Preflight Diagnostic
# ============================


from src.modules.utils.checkpoint_validator import validate_rrdb_checkpoint
from src.config.config_paths import UPSCALE_MODEL_PATHS
print("=== 🔍 Validating upscale model checkpoints ===")
for model_name, path in UPSCALE_MODEL_PATHS.items():
	if not path.exists():
		print(f"[Audit ❌] Missing checkpoint → {path}")
		continue

	valid = validate_rrdb_checkpoint(str(path))
	if not valid:
		print(f"[Audit ⚠️] Model '{model_name}' may be incompatible with RRDBNet")




from pathlib import Path
from datetime import datetime
from init_sweep import ensure_init_files

import os
import shutil


UPSCALE_MODEL_PATHS = {
	"UltraSharp": Path("models/Upscaler/upscaler-ultra/4x-UltraSharp.pth"),
	"Remacri": Path("models/Upscaler/upscaler-ultra/4x_foolhardy_Remacri.pth"),
	"Anime6B": Path("models/Upscaler/upscaler-ultra/RealESRGAN_x4plus_anime_6B.pth")
}

UPSCALE_MODELS = {
	"UltraSharp": "UltraSharp.pth",
	"Remacri": "Remacri.pth",
	"Anime6B": "Anime6B.pth"
}


print("[Debug] ✅ Running active preflight_check.py")
print("[Debug] 📍 Path:", __file__)

import torch
import os
import json
import subprocess
from datetime import datetime
import sys
from src.config.config_paths import UPSCALE_DIR

def load_core_manifest(manifest_path):
	with open(manifest_path, 'r') as f:
		return json.load(f)

def validate_core_files(manifest):
	missing = []
	for path in manifest['core_files'] + manifest['cosmos_modules']:
		full_path = os.path.join("F:\\SoftwareDevelopment\\AI Models Image\\AIGenerator", path)
		if not os.path.isfile(full_path):
			missing.append(full_path)
	return missing

# Usage
manifest_path = "F:\\SoftwareDevelopment\\AI Models Image\\AIGenerator\\config\\core_manifest.json"
manifest = load_core_manifest(manifest_path)
missing_files = validate_core_files(manifest)

if missing_files:
	print("🚨 Missing critical files:")
	for file in missing_files:
		print(f" - {file}")
else:
	print("✅ All core files present and accounted for.")


def purge_pycache(root="./src"):
	for dirpath, dirnames, filenames in os.walk(root):
		for dirname in dirnames:
			if dirname == "__pycache__":
				shutil.rmtree(os.path.join(dirpath, dirname), ignore_errors=True)

def validate_init_files(root="./src"):
	for dirpath, dirnames, filenames in os.walk(root):
		if "__init__.py" not in filenames:
			open(os.path.join(dirpath, "__init__.py"), "a").close()


def run_preflight():
	print("=== 🧠 SDXL Cockpit Preflight Check ===")

	# === Pycache Purge ===
	from init_sweep import purge_pycache
	purge_pycache("F:/SoftwareDevelopment/AI Models Image/AIGenerator")
	log_event({
		"event": "PycachePurge",
		"status": "complete",
		"timestamp": datetime.now().isoformat()
	})

def sanitize_event(event):
	if isinstance(event, dict):
		return {k: str(v) if isinstance(v, os.PathLike) else v for k, v in event.items()}
	return event

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# === CONFIG ===
VERBOSE_LOGGING = True
LOG_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/logs/telemetry_log.jsonl"
GUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gui", "main_window.py")

MODEL_PATHS = {
	"SDXL 1.0": "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/sdxl-base-1.0",
	"SDXL 1.5": "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/sdxl-base-1.5",
	"Refiner": "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/sdxl-refiner-1.0",
	"DreamShaper XL Turbo v2": "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_image/dreamshaper-xl-v2-turbo",
	"ComfyUI": "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models/text_to_video/ComfyUI"
}
# === UPSCALE MODELS ===
from src.config.config_paths import UPSCALE_MODEL_PATHS


def log_event(event):
	if not os.path.exists(LOG_PATH):
		with open(LOG_PATH, "w") as f:
			f.write("")
	safe_event = sanitize_event(event)
	with open(LOG_PATH, "a") as f:
		if isinstance(safe_event, dict):
			f.write(json.dumps(safe_event) + "\n")
		else:
			f.write(json.dumps({"message": str(safe_event), "timestamp": datetime.now().isoformat()}) + "\n")

def verbose_log(event):
	if not VERBOSE_LOGGING:
		return
	if not os.path.exists(LOG_PATH):
		with open(LOG_PATH, "w") as f:
			f.write("")
	safe_event = sanitize_event(event)
	with open(LOG_PATH, "a") as f:
		if isinstance(safe_event, dict):
			f.write("[VERBOSE] " + json.dumps(safe_event) + "\n")
		else:
			f.write("[VERBOSE] " + json.dumps({"message": str(safe_event), "timestamp": datetime.now().isoformat()}) + "\n")

def check_cuda():
	status = torch.cuda.is_available()
	device = torch.cuda.get_device_name(0) if status else "CPU"
	log_event({
		"event": "CUDA Check",
		"available": status,
		"device": device,
		"timestamp": datetime.now().isoformat()
	})
	print(f"[CUDA] Available: {status} | Device: {device}")

def check_vram():
	if torch.cuda.is_available():
		device = torch.cuda.current_device()
		props = torch.cuda.get_device_properties(device)
		log_event({
			"event": "VRAMStatus",
			"device": props.name,
			"total_vram_MB": props.total_memory // (1024 * 1024),
			"timestamp": datetime.now().isoformat()
		})
		print(f"[VRAM] {props.name}: {props.total_memory // (1024 * 1024)} MB")
	else:
		print("[VRAM] ❌ CUDA not available")

def check_dependencies():
	required = {
		"torch": "2.1.0",
		"transformers": "4.35.0",
		"diffusers": "0.24.0"
	}
	for pkg, expected in required.items():
		try:
			mod = __import__(pkg)
			version = getattr(mod, "__version__", "unknown")
			status = "✅" if version == expected else "⚠️"
			log_event({
				"event": "DependencyCheck",
				"package": pkg,
				"version": version,
				"expected": expected,
				"timestamp": datetime.now().isoformat()
			})
			print(f"[Dependency] {pkg}: {status} {version}")
		except Exception as e:
			print(f"[Dependency] ❌ {pkg} not found")

def check_model_paths():
	for name, path in MODEL_PATHS.items():
		exists = os.path.exists(path)
		verbose_log({
			"event": "ModelPathVerbose",
			"model": name,
			"path": path,
			"exists": exists,
			"timestamp": datetime.now().isoformat()
		})
		status = "✅ Found" if exists else "❌ Missing"
		print(f"[Model] {name}: {status}")

def check_refiner():
	try:
		from src.modules.refiner_module import get_refiner_status, load_refiner
		status = get_refiner_status()
		log_event({
			"event": "RefinerStatus",
			"available": status["available"],
			"device": status["device"],
			"path": status["path"],
			"timestamp": status["timestamp"]
		})

		if status["available"]:
			try:
				refiner = load_refiner(status["device"])
				print("[Refiner] ✅ Operational")
				log_event({
					"event": "RefinerLoad",
					"status": "success",
					"device": status["device"],
					"timestamp": datetime.now().isoformat()
				})
			except Exception as e:
				verbose_log({
					"event": "RefinerLoadFailed",
					"device": status["device"],
					"error": str(e),
					"timestamp": datetime.now().isoformat()
				})
				try:
					refiner = load_refiner("cpu")
					print("[Refiner] ⚠️ Retried with CPU fallback")
					log_event({
						"event": "RefinerLoad",
						"status": "cpu_fallback",
						"timestamp": datetime.now().isoformat()
					})
				except Exception as e:
					log_event({
						"event": "RefinerRetryFailed",
						"error": str(e),
						"timestamp": datetime.now().isoformat()
					})
					print("[Refiner] ❌ Retry failed")
		else:
			print("[Refiner] ❌ Unavailable")
	except Exception as e:
		log_event({
			"event": "RefinerImportError",
			"error": str(e),
			"timestamp": datetime.now().isoformat()
		})
		print("[Refiner] ❌ Import failed")




def check_gui_path():
	if not os.path.exists(GUI_PATH):
		log_event({
			"event": "GUIPathMissing",
			"path": GUI_PATH,
			"timestamp": datetime.now().isoformat()
		})
		print(f"[GUI] ❌ main_window.py not found at expected path:\n{GUI_PATH}")
		sys.exit(1)
	else:
		log_event({
			"event": "GUIPathCheck",
			"path": GUI_PATH,
			"exists": True,
			"timestamp": datetime.now().isoformat()
		})
		print("[GUI] ✅ main_window.py located successfully.")
		

def check_upscale_models():
	print("=== 🔼 Upscale Model Check ===")
	for name, path in UPSCALE_MODEL_PATHS.items():
		resolved_path = str(path.resolve())  # ✅ Absolute path
		exists = os.path.exists(resolved_path)
		status = "✅ Found" if exists else "❌ Missing"
		log_event({
			"event": "UpscaleModelCheck",
			"model": name,
			"path": resolved_path,
			"exists": exists,
			"timestamp": datetime.now().isoformat()
		})
		print(f"[Upscale] {name}: {status}")




def run_preflight():
	print("=== 🧠 SDXL Cockpit Preflight Check ===")

# === Version Telemetry Audit ===
version_check_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version_check.py")
try:
	subprocess.run(["python", version_check_path], check=True)
except subprocess.CalledProcessError as e:
	log_event({
		"event": "VersionCheckFailed",
		"error": str(e),
		"timestamp": datetime.now().isoformat()
	})
	print(f"[VersionCheck ❌] Failed: {e}")



	check_cuda()
	check_model_paths()
	check_refiner()
	check_gui_path()

	print("\n=== 🔼 Upscale Model Check ===")  # ✅ Add this banner
	check_upscale_models()                     # ✅ Call the function

	print("=== ✅ Preflight Complete ===")



# === Diagnostic Import Test ===
try:
	from src.modules import refiner_module
	print("[Refiner] ✅ Module imported directly")
except Exception as e:
	print("[Refiner] ❌ Direct import failed:", e)

if __name__ == "__main__":
	run_preflight()

