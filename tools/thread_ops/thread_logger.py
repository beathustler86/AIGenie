# thread_logger.py
from datetime import datetime

def log_event(thread_name, event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LOG] {timestamp} | {thread_name} | {event}")