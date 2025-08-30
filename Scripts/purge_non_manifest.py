import json
from pathlib import Path

# 🔒 Dry-run toggle
dry_run = True  # Set to False to enable actual deletion

# Load manifest
manifest_path = Path("F:/SoftwareDevelopment/AI Models Image/AIGenerator/config/core_manifest.json")
with open(manifest_path, "r") as f:
    manifest = json.load(f)

# Base directory
base_dir = Path("F:/SoftwareDevelopment/AI Models Image/AIGenerator").resolve()
allowed = set()

# ✅ Add manifest-tracked files
for key in ["core_files", "cosmos_modules", "assets"]:
    for rel_path in manifest.get(key, []):
        allowed.add((base_dir / rel_path).resolve())

# ✅ Add manifest-tracked folders
for folder_key in ["utils", "logs"]:
    for folder in manifest.get(folder_key, []):
        folder_path = (base_dir / folder).resolve()
        allowed.update(p.resolve() for p in folder_path.rglob("*") if p.is_file())

# ✅ Add cockpit-critical safe folders manually
safe_folders = ["venc", "tools", "telemetry", "outputs/refined"]
for folder in safe_folders:
    folder_path = (base_dir / folder).resolve()
    if folder_path.exists():
        allowed.update(p.resolve() for p in folder_path.rglob("*") if p.is_file())

# ✅ Build exclusion list from manifest
excluded_folders = [
    (base_dir / entry["folder"]).resolve()
    for entry in manifest.get("excluded_folders", [])
    if not entry.get("included", True)
]

# Scan all files in base_dir
all_files = [p.resolve() for p in base_dir.rglob("*") if p.is_file()]

# Sentinel files to preserve
sentinel_files = {".keep", ".gitkeep", ".placeholder"}

# Identify files to delete
files_to_delete = [
    p for p in all_files
    if p not in allowed and not any(str(p).startswith(str(ex)) for ex in excluded_folders)
]

# 🔥 Delete or simulate deletion
for file_path in files_to_delete:
    if file_path.name in sentinel_files:
        print(f"Skipping sentinel file: {file_path}")
        continue

    if dry_run:
        print(f"[DRY RUN] Would delete: {file_path}")
    else:
        print(f"Deleting: {file_path}")
        file_path.unlink()