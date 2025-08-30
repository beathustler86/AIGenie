import subprocess
import os
from datetime import datetime

# 🔧 CONFIGURATION
root_dir = r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image"
scripts = [
    "smart_archiver.py",
    "tree_profiler.py",
    "auto_archiver.py",
    "tree_analysis.py",
    "folder_archiver.py"
]
log_path = os.path.join(root_dir, r"outputs\logs\debug\optimizer_run_log.txt")

# 📝 LOGGING SETUP
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

# 🚀 EXECUTE SCRIPTS
log("🔧 Workspace Optimization Started")
for script in scripts:
    script_path = os.path.join(root_dir, "src", script)
    if os.path.exists(script_path):
        log(f"▶ Running {script}")
        try:
            subprocess.run(["python", script_path], check=True)
            log(f"✅ Completed {script}")
        except subprocess.CalledProcessError as e:
            log(f"❌ Error running {script}: {e}")
    else:
        log(f"⚠️ Script not found: {script}")

log("🏁 Optimization Complete")