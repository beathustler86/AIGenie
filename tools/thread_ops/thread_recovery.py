# thread_recovery.py
import threading
import time
from thread_logger import log_event

def resilient_thread(target, name="ResilientThread", retry_delay=2):
    def wrapper():
        while True:
            try:
                log_event(name, "Thread starting")
                target()
                log_event(name, "Thread completed")
                break
            except Exception as e:
                log_event(name, f"Crash detected — {e}")
                time.sleep(retry_delay)
                log_event(name, "Retrying...")
    thread = threading.Thread(target=wrapper, name=name, daemon=True)
    thread.start()
    log_event(name, "Resilient launch initiated")