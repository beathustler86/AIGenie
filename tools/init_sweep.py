import os

def ensure_init_files(root_dir):
	missing = []
	for dirpath, dirnames, filenames in os.walk(root_dir):
		# Skip hidden/system folders
		if any(part.startswith('.') for part in dirpath.split(os.sep)):
			continue
		# Check if folder contains Python files
		if any(fname.endswith('.py') for fname in filenames):
			init_path = os.path.join(dirpath, '__init__.py')
			if not os.path.exists(init_path):
				missing.append(init_path)
				# Auto-create blank __init__.py
				with open(init_path, 'w') as f:
					pass
				print(f"[Init ✅] Created: {init_path}")
	return missing

# 🔧 Run from cockpit root
missing_inits = ensure_init_files("F:/SoftwareDevelopment/AI Models Image/AIGenerator")
print(f"[Audit] Total missing __init__.py files patched: {len(missing_inits)}")