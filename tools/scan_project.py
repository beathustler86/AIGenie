import os

root_dir = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image"
log_path = os.path.join(root_dir, "outputs", "logs", "debug", "file_inventory.txt")

with open(log_path, "w", encoding="utf-8") as log:
    for folder, _, files in os.walk(root_dir):
        for file in files:
            full_path = os.path.join(folder, file)
            size = os.path.getsize(full_path)
            log.write(f"{file}\t{full_path}\t{size} bytes\n")

print(f"Inventory saved to: {log_path}")