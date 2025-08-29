import os
import json

MODEL_DIR = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/models"
SIZE_THRESHOLD_MB = 100
log_path = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/logs/model_log.json"

def scan_large_models(folder, threshold_mb):
    large_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > threshold_mb:
                large_files.append({
                    "file": path,
                    "size_MB": round(size_mb, 2)
                })
    return large_files

if __name__ == "__main__":
    results = scan_large_models(MODEL_DIR, SIZE_THRESHOLD_MB)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Logged {len(results)} large model files to {log_path}")
