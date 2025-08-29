
# ============================
# 🧠 launch_gui.py — SDXL Cockpit Launcher
# ============================
import subprocess
import os
import sys
sys.path.append(r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\src")

# ✅ Make ComfyUI folder importable for Cosmos checkpoint compatibility
comfy_path = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\models\text_to_video\ComfyUI"
if comfy_path not in sys.path:
    sys.path.append(comfy_path)



# ✅ Patch Python path to treat 'src' as root
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
	sys.path.append(project_root)


# ✅ Cockpit-native imports

from src.gui.main_window import MainWindow
from nodes.cosmos_text_to_video import CosmosTextToVideo
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
	sys.path.append(src_path)



from datetime import datetime

# Run preflight before GUI launch


def log(msg):
	print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_preflight_check():
	try:
		from src.modules import preflight_check
		preflight_check.run_preflight()
	except Exception as e:
		log(f"[Launcher] Preflight check failed: {e}")
		sys.exit(1)


run_preflight_check()


# === Launch GUI
import tkinter as tk

if __name__ == "__main__":
	log("[Launcher] 🚀 Cockpit boot sequence initiated")
	root = tk.Tk()
	root.geometry("2380x1220")
	root.configure(bg="#1e1e1e")
# === RESIZE WINDOW ===	root.resizable(False, False)



# === SD3.5 Loader Initialization ===
	try:
		from src.models.sd3_5_tensorrt.sd35_loader import initialize_sd35_modules
		sd35_sessions = initialize_sd35_modules()
		log("[SD3.5] ✅ Modules initialized")
	except Exception as e:
		sd35_sessions = {}
		log(f"[Launcher] SD3.5 loader failed: {e}")


	app = MainWindow(root, sd35_sessions=sd35_sessions)

	# === Telemetry Overlay Update ===
	try:
		app.update_telemetry_status()
	except Exception as e:
		log(f"[Launcher] Telemetry overlay failed: {e}")

	root.mainloop()