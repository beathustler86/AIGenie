import os
from collections import defaultdict
from datetime import datetime

# 🔧 CONFIGURATION
root_dir = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image"
summary_path = os.path.join(root_dir, r"outputs\logs\summary_tree_snapshot.txt")
top_files_path = os.path.join(root_dir, r"outputs\logs\summary_top_files.txt")
size_threshold_bytes = 10 * 1024 * 1024  # Flag files >10MB

# 📦 DATA STRUCTURES
folder_stats = defaultdict(lambda: {"count": 0, "size": 0})
extension_stats = defaultdict(int)
large_files = []

# 🧠 WALK TREE
for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            full_path = os.path.join(dirpath, file)
            size = os.path.getsize(full_path)
            ext = os.path.splitext(file)[-1].lower().lstrip(".")
            rel_folder = os.path.relpath(dirpath, root_dir)

            folder_stats[rel_folder]["count"] += 1
            folder_stats[rel_folder]["size"] += size
            extension_stats[ext] += 1

            if size >= size_threshold_bytes:
                large_files.append((file, full_path, size))
        except Exception:
            continue

# 📝 WRITE TREE SUMMARY
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"📁 Tree Snapshot Report\n")
    f.write(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("📂 Folder Stats:\n")
    for folder, stats in sorted(folder_stats.items(), key=lambda x: x[1]["size"], reverse=True):
        size_mb = stats["size"] / (1024**2)
        f.write(f"  - {folder}: {stats['count']} files, {size_mb:.2f} MB\n")

    f.write("\n📄 Extension Breakdown:\n")
    for ext, count in sorted(extension_stats.items(), key=lambda x: x[1], reverse=True):
        f.write(f"  .{ext}: {count} files\n")

# 📝 WRITE TOP FILES REPORT
os.makedirs(os.path.dirname(top_files_path), exist_ok=True)
with open(top_files_path, "w", encoding="utf-8") as f:
    f.write(f"📦 Large Files Report (>{size_threshold_bytes / (1024**2):.1f} MB)\n")
    f.write(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    for name, path, size in sorted(large_files, key=lambda x: x[2], reverse=True)[:50]:
        f.write(f"{name}\t{size / (1024**2):.2f} MB\t{path}\n")

# ✅ CONSOLE OUTPUT
print(f"\n📊 Tree summary saved to: {summary_path}")
print(f"📦 Top 50 large files saved to: {top_files_path}")