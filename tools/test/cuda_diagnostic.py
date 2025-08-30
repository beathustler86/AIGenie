# ============================
# ⚡ cuda_diagnostic.py — Full CUDA System Check
# ============================

import torch
import subprocess
import json
from datetime import datetime

LOG_PATH = "telemetry_log.jsonl"

def log_event(event):
	with open(LOG_PATH, "a") as f:
		if isinstance(event, dict):
			f.write(json.dumps(event) + "\n")
		else:
			f.write(json.dumps({"message": event, "timestamp": datetime.now().isoformat()}) + "\n")

def check_cuda():
	status = torch.cuda.is_available()
	device_count = torch.cuda.device_count()
	device_name = torch.cuda.get_device_name(0) if status else "CPU"
	capability = torch.cuda.get_device_capability(0) if status else "N/A"
	mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2 if status else 0

	log_event({
		"event": "CUDA Diagnostic",
		"available": status,
		"device_count": device_count,
		"device_name": device_name,
		"capability": capability,
		"memory_MB": round(mem_total, 2),
		"timestamp": datetime.now().isoformat()
	})

	print("=== ⚡ CUDA Diagnostic ===")
	print(f"[CUDA] Available: {status}")
	print(f"[CUDA] Device Count: {device_count}")
	print(f"[CUDA] Device Name: {device_name}")
	print(f"[CUDA] Compute Capability: {capability}")
	print(f"[CUDA] Total Memory: {mem_total:.2f} MB")

def check_driver():
	try:
		output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
		log_event({
			"event": "NVIDIA Driver Check",
			"output": output,
			"timestamp": datetime.now().isoformat()
		})
		print("=== 🧠 NVIDIA Driver Info ===")
		print(output)
	except Exception as e:
		log_event({
			"event": "NVIDIA Driver Check",
			"error": str(e),
			"timestamp": datetime.now().isoformat()
		})
		print("[Driver] ❌ nvidia-smi not available")

def run_cuda_diagnostic():
	check_cuda()
	check_driver()
	print("=== ✅ CUDA Diagnostic Complete ===")

if __name__ == "__main__":
	run_cuda_diagnostic()