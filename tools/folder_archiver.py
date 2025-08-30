import os
import shutil
import logging
from datetime import datetime

# 🔧 CONFIGURATION
root_dir = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image"
analysis_path = os.path.join(root_dir, r"outputs\logs\debug\tree_analysis.txt")
archive_root = os.path.join(root_dir, r"outputs\logs\archive\folders")
log_path = os.path.join(root_dir, r"outputs\logs\debug\folder_archive_log.txt")

size_threshold_mb = 100  # Only archive folders >100MB
depth_threshold = 3      # Only archive folders ≥3 levels deep

# 📝 LOGGING SETUP
os.makedirs(archive_root, exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

# 🧠 PARSE ANALYSIS
folders_to_archive = []

with open(analysis_path, "r", encoding="utf-8") as f:
    in_section = False
    for line in f:
        if line.strip().startswith("📁 Deep Folder Bloat"):
            in_section = True
            continue
        if in_section and line.strip().startswith("- "):
            try:
                parts = line.strip()[2:].split(":")
                folder = parts[0].strip()
                count_part, size_part = parts[1].split(",")
                size_mb = float(size_part.strip().split()[0])
                depth = folder.count(os.sep)
                if size_mb >= size_threshold_mb and depth >= depth_threshold:
                    folders_to_archive.append((folder, size_mb))
            except Exception:
                continue

# 🚚 MOVE FOLDERS
archived = []
for folder, size_mb in folders_to_archive:
    src_path = os.path.join(root_dir, folder)
    dest_path = os.path.join(archive_root, os.path.basename(folder))
    try:
        shutil.move(src_path, dest_path)
        logging.info(f"Archived folder: {folder} ({size_mb:.2f} MB)")
        archived.append((folder, size_mb))
    except Exception as e:
        logging.error(f"Failed to archive {folder}: {e}")

# ✅ CONSOLE OUTPUT
print(f"\n📦 Archived {len(archived)} folders to: {archive_root}")
for folder, size in archived[:10]:
    print(f"  - {folder} ({size:.2f} MB)")
print(f"\n📝 Log saved to: {log_path}")