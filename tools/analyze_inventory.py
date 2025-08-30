from collections import defaultdict
import os

inventory_path = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image\outputs\logs\debug\file_inventory.txt"
file_types = defaultdict(int)
total_size = 0

with open(inventory_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            name, path, size_str = parts
            ext = name.split(".")[-1].lower()
            try:
                size = int(size_str.replace(" bytes", ""))
                file_types[ext] += 1
                total_size += size
            except ValueError:
                continue

print("📦 File Type Breakdown:")
for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  .{ext}: {count} files")

print(f"\n🧮 Total Size: {total_size / (1024**2):.2f} MB")