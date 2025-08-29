# ============================
# 🧠 scan_refiner_file.py — Direct File Scanner
# ============================

import os

base_path = os.path.join(
	os.environ["USERPROFILE"],
	"AppData", "Local", "Programs", "Python", "Python310",
	"lib", "site-packages", "diffusers", "pipelines", "stable_diffusion_xl"
)

target_file = os.path.join(base_path, "refiner_pipeline.py")

if os.path.exists(target_file):
	print(f"✅ Found: {target_file}")
	with open(target_file, "r", encoding="utf-8") as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if "class StableDiffusionXLRefinerPipeline" in line:
				print(f"🔍 Line {i+1}: {line.strip()}")
else:
	print("❌ Refiner pipeline file not found.")