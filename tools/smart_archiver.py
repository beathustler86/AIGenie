import os
import shutil
import logging
from datetime import datetime

# 🔧 CONFIGURATION
inventory_path = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\outputs\logs\debug\file_inventory.txt"
archive_dir = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\archive"
log_path = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\outputs\logs\debug\archive_log.txt"
summary_path = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\outputs\logs\summary_archived_files.txt"

extensions_to_archive = {
    "pyc", "pyd", "dll", "exe", "lib", "a", "obj", "so", "whl", "wheel",
    "testcase", "tmp", "cache", "lock", "zip-safe", "orig",
    "license", "notice", "metadata", "record", "authors", "copying",
    "gmt", "utc", "tokyo", "london", "new_york", "gmt+0", "gmt-0"
}

size_threshold_bytes = 0  # Set >0 to archive only files above a certain size
dry_run = True            # Set to False to actually move files

# 📝 LOGGING SETUP
os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(archive_dir, exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

# 🧠 PROCESSING
archived = []

def move_file(src, dest):
    if dry_run:
        logging.info(f"[DRY-RUN] Would move: {src} → {dest}")
    else:
        try:
            shutil.move(src, dest)
            logging.info(f"Moved: {src} → {dest}")
        except Exception as e:
            logging.error(f"Failed to move {src}: {e}")

with open(inventory_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        name, path, size_str = parts
        ext = name.split(".")[-1].lower()
        try:
            size = int(size_str.replace(" bytes", ""))
        except ValueError:
            continue

        if ext in extensions_to_archive and size >= size_threshold_bytes:
            dest_path = os.path.join(archive_dir, os.path.basename(path))
            move_file(path, dest_path)
            archived.append((name, size))

# 📊 SUMMARY REPORT
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as report:
    report.write(f"📦 {'Previewed' if dry_run else 'Archived'} {len(archived)} files\n")
    report.write(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    for name, size in archived:
        report.write(f"{name}\t{size / 1024:.1f} KB\n")

# ✅ CONSOLE OUTPUT
print(f"\n📦 {'Previewed' if dry_run else 'Archived'} {len(archived)} files to {archive_dir}")
for name, size in archived[:10]:
    print(f"  - {name} ({size / 1024:.1f} KB)")
print(f"\n📝 Log saved to: {log_path}")
print(f"📊 Summary saved to: {summary_path}")