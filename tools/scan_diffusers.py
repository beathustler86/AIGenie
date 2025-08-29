# ============================
# 🧠 scan_diffusers.py — Refiner Class Scanner
# ============================

import os
import inspect
import diffusers

def scan_module(module):
	found = []
	for name, obj in inspect.getmembers(module):
		if inspect.isclass(obj) and "Refiner" in name:
			found.append(name)
	return found

print("=== 🔍 Scanning diffusers ===")
top_level = scan_module(diffusers)
print(f"[Top Level] {top_level}")

try:
	from diffusers.pipelines import stable_diffusion_xl
	sub_level = scan_module(stable_diffusion_xl)
	print(f"[stable_diffusion_xl] {sub_level}")
except Exception as e:
	print(f"[Error] Could not scan stable_diffusion_xl: {e}")