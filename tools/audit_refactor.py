import os

ROOT = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\src"
LOG_FILE = os.path.join(ROOT, "workspace_inventory.txt")

# === Expected Files by Refactor Plan ===
expected = {
    "gui/main_window.py",
    "gui/gradio_interface.py",
    "pipeline/sdxl_base_refiner.py",
    "pipeline/benchmark_runner.py",
    "pipeline/benchmark_runner2.py",
    "archiver/archive_inventory.py",
    "archiver/auto_archiver.py",
    "archiver/folder_archiver.py",
    "archiver/smart_archiver.py",
    "audit/analyze_inventory.py",
    "audit/scan_project.py",
    "audit/tree_analysis.py",
    "audit/tree_profiler.py",
    "audit/workspace_optimizer.py",
    "main.py"
}

# === Scan Workspace ===
found = set()
with open(LOG_FILE, "w", encoding="utf-8") as log:
    for dirpath, _, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        log.write(f"\n📁 {rel_dir}\n")
        for file in filenames:
            rel_path = os.path.normpath(os.path.join(rel_dir, file))
            found.add(rel_path)
            log.write(f" - {file}\n")

# === Compare Results ===
print("\n📦 Refactor Audit Summary:")
for path in sorted(expected):
    if path in found:
        print(f"✅ Found: {path}")
    else:
        print(f"❌ Missing: {path}")

extra = found - expected
if extra:
    print("\n🧩 Extra Files Detected:")
    for path in sorted(extra):
        print(f" - {path}")

print(f"\n🧾 Full inventory saved to:\n{LOG_FILE}")