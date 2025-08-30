import tkinter as tk
from gui.widgets.thread_status_widget import ThreadStatusWidget
from gui.widgets.telemetry_overlay import TelemetryOverlay
from gui.widgets.log_viewer import LogViewer

class ControlPanel(tk.Frame):
    def __init__(self, master=None, controller=None):
        super().__init__(master)
        self.controller = controller
        self.pack(fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        # Thread Status Zone
        self.thread_status = ThreadStatusWidget(self)
        self.thread_status.pack(side="top", fill="x", padx=10, pady=5)

        # Telemetry Overlay Zone
        self.telemetry_overlay = TelemetryOverlay(self)
        self.telemetry_overlay.pack(side="top", fill="x", padx=10, pady=5)

        # Log Viewer Zone
        self.log_viewer = LogViewer(self)
        self.log_viewer.pack(side="bottom", fill="both", expand=True, padx=10, pady=5)

        # === Tactical Controls Zone ===
        controls_frame = tk.LabelFrame(self, text="🎛️ Tactical Controls", padx=10, pady=10)
        controls_frame.pack(side="top", fill="x", padx=10, pady=5)

        self.strength_slider = tk.Scale(controls_frame, from_=0, to=100, orient="horizontal", label="Strength")
        self.strength_slider.set(50)
        self.strength_slider.pack(side="left", padx=5)

        self.detail_slider = tk.Scale(controls_frame, from_=0, to=100, orient="horizontal", label="Detail")
        self.detail_slider.set(75)
        self.detail_slider.pack(side="left", padx=5)

        generate_btn = tk.Button(controls_frame, text="🚀 Generate", command=self.trigger_generate)
        generate_btn.pack(side="left", padx=5)

    def trigger_generate(self):
        strength = self.strength_slider.get()
        detail = self.detail_slider.get()
        print(f"🚀 Generate triggered — Strength: {strength}, Detail: {detail}")
        if self.controller:
            self.controller.threaded_generate(strength, detail)