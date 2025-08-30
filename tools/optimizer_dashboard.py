import tkinter as tk
from tkinter import scrolledtext, BooleanVar
import subprocess
import os
import time
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
script_paths = [os.path.join(root_dir, "src", s) for s in scripts]
status_labels = {}

# 🧠 GUI SETUP
window = tk.Tk()
window.title("Optimizer Dashboard")
window.geometry("900x650")
window.configure(bg="#1e1e1e")

# 📝 Console Output
console = scrolledtext.ScrolledText(window, wrap=tk.WORD, bg="#252526", fg="#d4d4d4", font=("Consolas", 10))
console.place(x=20, y=220, width=860, height=400)

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    console.insert(tk.END, f"[{timestamp}] {message}\n")
    console.see(tk.END)

def update_status(script, status, duration=None):
    label = status_labels.get(script)
    if label:
        if status == "success":
            label.config(text=f"🟢 Completed ({duration:.1f}s)", fg="#6A9955")
        elif status == "error":
            label.config(text="🔴 Error", fg="#F44747")
        elif status == "running":
            label.config(text="⏳ Running...", fg="#DCDCAA")

def run_script(script_path, script_name):
    if not os.path.exists(script_path):
        log(f"⚠️ Script not found: {script_name}")
        update_status(script_name, "error")
        return
    log(f"▶ Running {script_name}")
    update_status(script_name, "running")
    start = time.time()
    try:
        subprocess.run(["python", script_path], check=True)
        duration = time.time() - start
        log(f"✅ Completed {script_name} in {duration:.1f}s")
        update_status(script_name, "success", duration)
    except subprocess.CalledProcessError as e:
        log(f"❌ Error running {script_name}: {e}")
        update_status(script_name, "error")

def run_all():
    log("🔧 Workspace Optimization Started")
    for i, path in enumerate(script_paths):
        run_script(path, scripts[i])
    log("🏁 Optimization Complete")

# 🎛️ Buttons + Status
button_frame = tk.Frame(window, bg="#1e1e1e")
button_frame.place(x=20, y=20)

for i, script in enumerate(scripts):
    script_name = script.replace(".py", "")
    btn = tk.Button(button_frame, text=f"Run {script_name}", width=25,
                    command=lambda p=script_paths[i], s=scripts[i]: run_script(p, s),
                    bg="#007acc", fg="white", font=("Segoe UI", 10))
    btn.grid(row=i, column=0, pady=5)

    status = tk.Label(button_frame, text="⏺ Idle", fg="#CCCCCC", bg="#1e1e1e", font=("Segoe UI", 9))
    status.grid(row=i, column=1, padx=10)
    status_labels[scripts[i]] = status

# 🧠 Run All Button
run_all_btn = tk.Button(window, text="Run Full Optimization", width=30, command=run_all,
                        bg="#0e639c", fg="white", font=("Segoe UI", 11, "bold"))
run_all_btn.place(x=600, y=30)

# 📝 Log Viewer Button
def open_log():
    log_path = os.path.join(root_dir, r"outputs\logs\debug\optimizer_run_log.txt")
    if os.path.exists(log_path):
        os.startfile(log_path)
    else:
        log("⚠️ Log file not found.")

log_btn = tk.Button(window, text="Open Optimizer Log", width=30, command=open_log,
                    bg="#3c3c3c", fg="white", font=("Segoe UI", 10))
log_btn.place(x=600, y=80)

# ⚙️ Dry-Run Toggle
dry_run_var = BooleanVar(value=True)

def toggle_dry_run():
    state = dry_run_var.get()
    log(f"⚙️ Dry-run mode {'enabled' if state else 'disabled'}")
    # Optional: write to config file or pass as env var

dry_run_check = tk.Checkbutton(window, text="Dry-run Mode", variable=dry_run_var,
                               command=toggle_dry_run, bg="#1e1e1e", fg="white",
                               font=("Segoe UI", 10), selectcolor="#1e1e1e")
dry_run_check.place(x=600, y=130)

window.mainloop()