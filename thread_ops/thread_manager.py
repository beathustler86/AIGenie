# thread_manager.py
import threading

def launch_thread(target, name="UnnamedThread"):
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    print(f"[THREAD] Launched: {name}")