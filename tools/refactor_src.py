import os
import shutil

# === Folder Setup ===
folders = [
    "gui", "pipeline", "archiver", "audit", "utils"
]
for folder in folders:
    path = f"src/{folder}"
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, "__init__.py"), "w").close()

# === File Moves ===
file_moves = {
    "generative_fill_gui.py": "gui/main_window.py",
    "sdxl_base_refiner.py": "pipeline/sdxl_base_refiner.py",
    "sdxl_benchmark.py": "pipeline/benchmark_runner.py",
    "sdxl_benchmark2.py": "pipeline/benchmark_runner2.py",
    "sdxl_gradio.py": "gui/gradio_interface.py",
    "archive_inventory,py.txt": "archiver/archive_inventory.py",
    "auto_archiver.py": "archiver/auto_archiver.py",
    "folder_archiver.py": "archiver/folder_archiver.py",
    "smart_archiver.py": "archiver/smart_archiver.py",
    "analyze_inventory.py": "audit/analyze_inventory.py",
    "scan_project.py": "audit/scan_project.py",
    "tree_analysis.py": "audit/tree_analysis.py",
    "tree_profiler.py": "audit/tree_profiler.py",
    "workspace_optimizer.py": "audit/workspace_optimizer.py"
}

log = []

for src_name, dst_rel in file_moves.items():
    src_path = f"src/{src_name}"
    dst_path = f"src/{dst_rel}"
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        log.append(f"Moved: {src_name} → {dst_rel}")
    else:
        log.append(f"Missing: {src_name} (skipped)")

# === Create main.py ===
main_path = "src/main.py"
if not os.path.exists(main_path):
    with open(main_path, "w") as f:
        f.write("from gui.main_window import launch_gui\n\nif __name__ == '__main__':\n    launch_gui()\n")
    log.append("Created: main.py")

# === Cleanup ===
junk = ["New Text Document.txt"]
for j in junk:
    j_path = f"src/{j}"
    if os.path.exists(j_path):
        os.remove(j_path)
        log.append(f"Deleted: {j}")

# === Summary ===
print("\n📦 Refactor Complete:")
for entry in log:
    print(" -", entry)