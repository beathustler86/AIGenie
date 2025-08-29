# ============================
# 🧠 launch_gui.py — SDXL Cockpit Launcher
# ============================
import subprocess
import os
import sys
import traceback
import tkinter as tk
from datetime import datetime

sys.path.append(r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\src")

from src.modules.preflight_check import (
	check_sdxl_1_0,
	check_dreamshaper_v2,
	check_comfyui
)

from src.gui.main_window import MainWindow
from nodes.cosmos_text_to_video import CosmosTextToVideo

# ✅ Make ComfyUI folder importable for Cosmos checkpoint compatibility
comfy_path = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_video\ComfyUI"
if comfy_path not in sys.path:
	sys.path.append(comfy_path)

# ✅ Patch Python path to treat 'src' as root
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
	sys.path.append(project_root)

def log(msg):
	print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Run preflight check
def run_preflight_check():
	try:
		from src.modules import preflight_check
		preflight_check.run_preflight()
	except Exception as e:
		log(f"[Launcher] Preflight check failed: {e}")
		traceback.print_exc()
		sys.exit(1)

run_preflight_check()

if __name__ == "__main__":
	log("[Launcher] 🚀 Cockpit boot sequence initiated")

	root = tk.Tk()
	root.geometry("2560x1440")
	root.configure(bg="#1e1e1e")
	root.resizable(True,True)

	# === SD3.5 Loader Initialization ===
	try:
		from src.models.sd3_5_tensorrt.sd35_loader import initialize_sd35_modules
		sd35_sessions = initialize_sd35_modules()
		log("[SD3.5] ✅ Modules initialized")
	except Exception as e:
		sd35_sessions = {}
		log(f"[Launcher] SD3.5 loader failed: {e}")

	# Create the main window once, with SD3.5 sessions
	app = MainWindow(root, sd35_sessions=sd35_sessions)

	# === Telemetry Overlay Update ===
	try:
		app.update_telemetry_status()
	except Exception as e:
		log(f"[Launcher] Telemetry overlay failed: {e}")

	app.mainloop()
