import os

def walk_dir(root, max_depth=2, exclude=("env", "__pycache__", "logs")):
    with open("cockpit_structure_trimmed.txt", "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip excluded folders
            if any(ex in dirpath.split(os.sep) for ex in exclude):
                continue
            depth = dirpath[len(root):].count(os.sep)
            if depth > max_depth:
                continue
            indent = '  ' * depth
            f.write(f"{indent}{os.path.basename(dirpath)}/\n")
            for file in filenames:
                f.write(f"{indent}  {file}\n")

walk_dir(r"F:\SoftwareDevelopment\AI Models Image\-AI_Models_Image")