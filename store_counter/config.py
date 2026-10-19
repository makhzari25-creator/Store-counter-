"""Configuration: config.json next to the exe (or next to main.py in dev)."""
import copy
import json
import os
import sys
import threading

DEFAULTS = {
    # "demo" | camera index (e.g. "0") | RTSP url | video file path
    # Hikvision: rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101
    "source": "demo",
    "host": "127.0.0.1",          # use 0.0.0.0 to open the panel to the LAN
    "port": 8080,
    "admin_password": "",         # empty = no login. User name: admin
    "open_browser": True,
    "data_dir": "data",
    "model_path": "models/yolov8n.onnx",
    "detector": "auto",           # auto | yolo | motion
    "conf": 0.35,
    "imgsz": 640,
    "process_fps": 10,
    # Counting line, fractions of frame width/height. A->B direction:
    # crossing from the left side of A->B to the right side counts as "in"
    "line": {"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5},
    "invert_direction": False,
    "min_track_hits": 3,
    "green": {                    # OpenCV HSV: H 0-179, S/V 0-255
        "h_min": 35, "h_max": 85, "s_min": 90, "v_min": 60,
        "min_ratio": 0.04,        # share of green pixels inside the person box
        "min_frame_fraction": 0.25,
    },
    "reset_inside_daily": True,   # occupancy counter resets at midnight
    "snapshot_interval": 60,      # seconds between occupancy snapshots
}


def app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def web_dir() -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "store_counter", "web")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        elif k in out:
            out[k] = v
    return out


class Config:
    def __init__(self, path: str = None):
        self.path = path or os.path.join(app_dir(), "config.json")
        self.lock = threading.RLock()
        self.data = copy.deepcopy(DEFAULTS)
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = _merge(DEFAULTS, json.load(f))
        else:
            self.save()

    def resolve(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(app_dir(), p)

    def __getitem__(self, k):
        return self.data[k]

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.data)

    def update(self, partial: dict):
        """Validate + merge a partial config coming from the panel."""
        with self.lock:
            new = _merge(self.data, partial)
            ln = new["line"]
            for k in ("x1", "y1", "x2", "y2"):
                ln[k] = min(1.0, max(0.0, float(ln[k])))
            if (ln["x1"], ln["y1"]) == (ln["x2"], ln["y2"]):
                raise ValueError("نقطه ابتدا و انتهای خط نباید یکی باشد")
            new["port"] = int(new["port"])
            new["conf"] = min(0.95, max(0.05, float(new["conf"])))
            new["process_fps"] = min(30, max(1, int(new["process_fps"])))
            new["invert_direction"] = bool(new["invert_direction"])
            new["source"] = str(new["source"]).strip() or "demo"
            g = new["green"]
            for k in ("h_min", "h_max", "s_min", "v_min"):
                g[k] = int(g[k])
            g["min_ratio"] = float(g["min_ratio"])
            g["min_frame_fraction"] = float(g["min_frame_fraction"])
            self.data = new
            self.save()

    def save(self):
        with self.lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
