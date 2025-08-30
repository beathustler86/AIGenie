import os
from datetime import datetime

def save_image(image, config):
    filename = config["filename"] or f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join("bin", filename)
    image.save(path)
    log_save(path)

def bin_image(config):
    filename = config["filename"] or "fault_image.png"
    path = os.path.join("bin", "faults", filename)
    # Save placeholder or error image
    log_fault(path)

def log_save(path):
    print(f"[AUDIT] Saved to {path}")

def log_fault(path):
    print(f"[AUDIT] Fault image routed to {path}")