import os
import shutil

# 🔧 CONFIGURATION
inventory_path = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\outputs\logs\debug\file_inventory.txt"
archive_dir = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\archive"
extensions_to_archive = {
    "pyc", "pyd", "dll", "exe", "lib", "a", "obj", "so", "whl", "wheel",
    "testcase", "tmp", "cache", "lock", "zip-safe", "orig",
    "license", "notice", "metadata", "record", "authors", "copying",
    "gmt", "utc", "tokyo", "london", "new_york", "gmt+0", "gmt-0"
}
size_threshold_bytes = 0  # Set >0 to archive only files above a certain size

# 🧠 PROCESSING
os.makedirs(archive_dir, exist_ok=True)
archived = []

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
            try:
                shutil.move(path, dest_path)
                archived.append((name, size))
            except Exception as e:
                print(f"⚠️ Could not move {path}: {e}")

# 📊 SUMMARY
print(f"\n📦 Archived {len(archived)} files to {archive_dir}")
for name, size in archived[:10]:  # Show first 10
    print(f"  - {name} ({size / 1024:.1f} KB)")