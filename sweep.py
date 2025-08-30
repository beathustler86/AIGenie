import os
import re

# 🔍 Target pattern
pattern = r"\.\s*Linear\s*\("

# 📁 Root directory to scan
root_dir = r"F:\SoftwareDevelopment\AI Models Image\AIGenerator\src"

# 📄 Results
matches = []

for dirpath, _, filenames in os.walk(root_dir):
	for filename in filenames:
		if filename.endswith(".py"):
			full_path = os.path.join(dirpath, filename)
			with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
				for i, line in enumerate(f):
					if re.search(pattern, line):
						matches.append((full_path, i + 1, line.strip()))