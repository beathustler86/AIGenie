from datetime import datetime
import diffusers, transformers, huggingface_hub

log_path = "F:/SoftwareDevelopment/AI Models Image/AIGenerator/src/modules/telemetry.log"

with open(log_path, "a") as log:
	log.write(f"\n[{datetime.now()}] Version Check:\n")
	log.write(f"  diffusers: {diffusers.__version__}\n")
	log.write(f"  transformers: {transformers.__version__}\n")
	log.write(f"  huggingface_hub: {huggingface_hub.__version__}\n")

print("[Telemetry] Version check logged.")
