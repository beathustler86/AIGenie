import json
import os
import subprocess
from datetime import datetime

LOG_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/logs/model_log.json"
TRACKED_FILES = ["config/core_manifest.json"]

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def get_latest_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "Unknown"

def get_manifest_diff(file_path):
    try:
        result = subprocess.run(
            ["git", "diff", "--unified=0", "HEAD~1", "HEAD", "--", file_path],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or "No changes detected"
    except subprocess.CalledProcessError:
        return "Diff unavailable"

def log_manifest_push(files, status="Tracked and pushed"):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": "Manifest push",
        "commit_hash": get_latest_commit(),
        "files": files,
        "status": status,
        "diff": {f: get_manifest_diff(f) for f in files}
    }

    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r+") as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=4)
            print("✅ Push event logged successfully.")
        else:
            with open(LOG_PATH, "w") as f:
                json.dump([entry], f, indent=4)
            print("🆕 Log file created and push event logged.")
    except Exception as e:
        print(f"❌ Logging failed: {e}")

    if os.path.getsize(LOG_PATH) > 5_000_000:
        print("⚠️ Log file size exceeds 5MB. Consider rotating or archiving.")

# 🔧 Trigger logging
log_manifest_push(TRACKED_FILES)
