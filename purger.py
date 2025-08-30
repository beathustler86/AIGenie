import shutil
import os

search_root = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\src"

for root, dirs, _ in os.walk(search_root):
	for d in dirs:
		if d == "__pycache__":
			cache_path = os.path.join(root, d)
			shutil.rmtree(cache_path)
			print(f"🧹 Purged: {cache_path}")