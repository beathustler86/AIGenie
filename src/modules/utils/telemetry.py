import os
import json
from datetime import datetime

LOG_PATH = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/logs/telemetry_log.jsonl"

def sanitize_event(event):
	if isinstance(event, dict):
		return {k: str(v) if isinstance(v, os.PathLike) else v for k, v in event.items()}
	return event

def log_event(event):
	if not os.path.exists(LOG_PATH):
		with open(LOG_PATH, "w") as f:
			f.write("")
	safe_event = sanitize_event(event)
	with open(LOG_PATH, "a") as f:
		if isinstance(safe_event, dict):
			f.write(json.dumps(safe_event) + "\n")
		else:
			f.write(json.dumps({"message": str(safe_event), "timestamp": datetime.now().isoformat()}) + "\n")