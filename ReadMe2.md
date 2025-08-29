# 🧠 AIGenerator — Modular AI Image & Video Cockpit

AIGenerator is a cockpit-grade Python application for orchestrating AI-powered image and video workflows. Built for modularity, auditability, and GUI control, it integrates SDXL-based pipelines, Cosmos video generation, telemetry overlays, and manifest-driven configuration.

---

## 🚀 Features

- **GUI Launcher** with fallback widgets and status telemetry  
- **Text-to-Image Generation** via SDXL and custom pipelines  
- **Text-to-Video Generation** using Cosmos modules and tokenizers  
- **SD3.5 TensorRT Loader** for optimized inference  
- **Modular Pipelines** for upscaling, refining, and benchmarking  
- **Audit Tools** for inventory scans, tree profiling, and workspace hygiene  
- **Manifest-Driven Config** for traceable deployments  
- **CUDA Diagnostics** and structure scans for preflight validation  
- **Telemetry Overlay** for real-time cockpit feedback

---

## 🧰 Project Structure

```plaintext
AIGenerator/
├── launch_gui.py              # Entry point for GUI cockpit
├── purger.py                  # Cleanup utility
├── init_sweep.py              # Initializer sweep logic
├── config/core_manifest.json  # Manifest of all critical modules
├── src/
│   ├── modules/               # Core logic modules
│   ├── nodes/                 # Model definitions and Cosmos logic
│   ├── models/                # SD3.5 TensorRT loaders
│   ├── gui/                   # GUI and widgets
│   ├── pipeline/              # Image and video processing pipelines
│   ├── audit/                 # Audit and profiling tools
│   ├── config/                # Config path logic
│   ├── data/                  # Static assets
├── test/                      # Diagnostics and validation scripts
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git hygiene
├── readme.md                  # This file
```

---

## 🎬 Text-to-Video & Text-to-Image

**Text-to-Image**  
- Powered by SDXL and custom refiner modules  
- Supports FP16 export and benchmarking  
- Pipeline: `src/pipeline/sdxl_base_refiner.py`

**Text-to-Video**  
- Cosmos-based architecture with 3D token embedding  
- Modules:  
  - `src/nodes/cosmos_text_to_video.py`  
  - `src/nodes/cosmos/model.py`  
  - `src/nodes/cosmos/cosmos_tokenizer/layers3d.py`

---

## ⚙️ Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/AIGenerator.git
   cd AIGenerator
   ```

2. **Create a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run cockpit GUI**  
   ```bash
   python launch_gui.py
   ```

---

## 🧪 Preflight Diagnostics

Run these before deployment:

```bash
python test/cuda_diagnostic.py
python test/structure_scan.py
python src/modules/preflight_check.py
```

---

## 📦 Manifest Logic

The `core_manifest.json` defines all mission-critical modules, assets, and config paths. It ensures traceable deployments and fallback logic for missing components.

---

## 📜 License

MIT — free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests welcome. For major changes, open an issue first to discuss what you’d like to change.