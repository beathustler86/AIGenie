import os
import site
import json
from datetime import datetime

LOG_PATH = "size_log.json"
SIZE_LIMIT_MB = 8192  # 8GB threshold

def get_total_package_size():
	total_size = 0
	site_packages = site.getsitepackages()[0]
	for item in os.listdir(site_packages):
		item_path = os.path.join(site_packages, item)
		if os.path.isdir(item_path):
			for root, _, files in os.walk(item_path):
				for f in files:
					fp = os.path.join(root, f)
					if os.path.isfile(fp):
						total_size += os.path.getsize(fp)
	return round(total_size / (1024 ** 2), 2)  # MB

def load_previous_log():
	if not os.path.exists(LOG_PATH):
		return None
	with open(LOG_PATH, "r") as f:
		try:
			return json.load(f)
		except json.JSONDecodeError:
			return None

def save_log(entry):
	with open(LOG_PATH, "w") as f:
		json.dump(entry, f, indent=4)

if __name__ == "__main__":
	current_size = get_total_package_size()
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	prev_log = load_previous_log()

delta = None
if prev_log:
	delta = round(current_size - prev_log["size_mb"], 2)

log_entry = {
	"timestamp": timestamp,
	"size_mb": current_size,
	"delta_mb": delta
}

save_log(log_entry)

print(f"📦 Total installed package size: {current_size} MB")
if delta is not None:
	print(f"🔄 Change since last audit: {delta:+} MB")
if current_size > SIZE_LIMIT_MB:
	print(f"⚠️ Warning: Package footprint exceeds {SIZE_LIMIT_MB} MB threshold")
