import os
import shutil
import logging
from datetime import datetime, timedelta

# 🔧 CONFIGURATION
root_dir = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image"
archive_dir = os.path.join(root_dir, r"outputs\logs\archive")
log_path = os.path.join(root_dir, r"outputs\logs\debug\auto_archive_log.txt")
age_threshold = timedelta(hours=2)

# 📝 LOGGING SETUP
os.makedirs(archive_dir, exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

# 🧠 PROCESSING
now = datetime.now()
archived = []

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        full_path = os.path.join(dirpath, file)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
            if now - mtime > age_threshold:
                rel_path = os.path.relpath(full_path, root_dir)
                dest_path = os.path.join(archive_dir, os.path.basename(full_path))
                shutil.move(full_path, dest_path)
                logging.info(f"Archived: {rel_path} → {dest_path}")
                archived.append((file, rel_path))
        except Exception as e:
            logging.error(f"Failed to archive {full_path}: {e}")

# ✅ CONSOLE OUTPUT
print(f"\n📦 Archived {len(archived)} files older than 2 hours to: {archive_dir}")
for name, rel in archived[:10]:
    print(f"  - {name} from {rel}")
print(f"\n📝 Log saved to: {log_path}")