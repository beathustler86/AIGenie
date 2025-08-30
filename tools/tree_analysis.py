import os
from collections import defaultdict
from datetime import datetime

# 🔧 CONFIGURATION
snapshot_path = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\outputs\logs\summary_tree_snapshot.txt"
output_path = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\outputs\logs\debug\tree_analysis.txt"

# 📦 DATA STRUCTURES
extension_counts = defaultdict(int)
folder_sizes = defaultdict(float)
deep_folders = []

# 🧠 PARSE SNAPSHOT
with open(snapshot_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("📁") or line.startswith("🕒"):
            continue
        if line.startswith("  - "):  # Folder stats
            try:
                parts = line[4:].split(":")
                folder = parts[0].strip()
                count_part, size_part = parts[1].split(",")
                count = int(count_part.strip().split()[0])
                size_mb = float(size_part.strip().split()[0])
                folder_sizes[folder] = size_mb
                if folder.count(os.sep) >= 3:
                    deep_folders.append((folder, count, size_mb))
            except Exception:
                continue
        elif line.startswith("  ."):  # Extension breakdown
            try:
                ext, count = line[3:].split(":")
                extension_counts[ext.strip()] = int(count.strip().split()[0])
            except Exception:
                continue

# 📝 WRITE ANALYSIS
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as out:
    out.write(f"📊 Tree Analysis Report\n")
    out.write(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    out.write("🔍 Top Extensions:\n")
    for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        out.write(f"  .{ext}: {count} files\n")

    out.write("\n📦 Largest Folders:\n")
    for folder, size in sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)[:20]:
        out.write(f"  - {folder}: {size:.2f} MB\n")

    out.write("\n📁 Deep Folder Bloat (≥3 levels):\n")
    for folder, count, size in sorted(deep_folders, key=lambda x: x[2], reverse=True)[:10]:
        out.write(f"  - {folder}: {count} files, {size:.2f} MB\n")

print(f"\n✅ Tree analysis saved to: {output_path}")