"""Background thread: camera -> detector -> engine -> database, plus MJPEG preview."""
import os
import threading
import time
import traceback

import cv2

from .demo import DemoDetector, SyntheticScene
from .detectors import build_detector
from .engine import CountingEngine

# RTSP over TCP is far more reliable than UDP on old Hikvision cameras / Wi-Fi
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class Pipeline(threading.Thread):
    def __init__(self, cfg, db, log=print):
        super().__init__(daemon=True)
        self.cfg, self.db, self.log = cfg, db, log
        self._stop = threading.Event()
        self._reload = threading.Event()
        self._cond = threading.Condition()
        self._jpeg = None
        self._frame_id = 0
        self._status = {"connected": False, "fps": 0.0, "in_frame": 0, "detector": "-",
                        "error": "", "source": cfg["source"]}

    # ---- public --------------------------------------------------------------
    def status(self) -> dict:
        s = dict(self._status)
        s["source_kind"] = "demo" if str(self.cfg["source"]) == "demo" else "camera"
        return s

    def reload(self):
        self._reload.set()

    def stop(self):
        self._stop.set()

    def wait_frame(self, last_id, timeout=2.0):
        with self._cond:
            if self._frame_id == last_id:
                self._cond.wait(timeout)
            return self._frame_id, self._jpeg

    # ---- thread --------------------------------------------------------------
    def run(self):
        while not self._stop.is_set():
            try:
                self._reload.clear()
                self._run_once()
            except Exception as e:                       # keep running, retry
                self._status.update(connected=False, error=str(e))
                self.log("[pipeline] error:\n" + traceback.format_exc())
                self._stop.wait(5)

    def _open(self, src):
        s = str(src)
        if s.isdigit():
            cap = cv2.VideoCapture(int(s), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(s))
        else:
            cap = cv2.VideoCapture(s)
        if not cap.isOpened():
            raise ConnectionError(f"cannot open camera/source: {s}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run_once(self):
        conf = self.cfg.snapshot()
        engine = CountingEngine(conf)
        src = str(conf["source"])
        db = self.db.for_source(src)          # demo data never touches the real database
        scene = cap = None
        if src == "demo":
            scene = SyntheticScene(seed=int(time.time()) % 1000)
            detector = DemoDetector()
        else:
            detector = build_detector(self.cfg, self.log)
            cap = self._open(src)
        self._status.update(connected=True, error="", detector=detector.name, source=src)
        self.log(f"[pipeline] started, source={'demo' if scene else 'camera'} detector={detector.name}")

        min_dt = 1.0 / conf["process_fps"]
        last_proc = last_snap = 0.0
        fps_n, fps_t0 = 0, time.time()
        fails = 0
        try:
            while not self._stop.is_set() and not self._reload.is_set():
                now = time.time()
                if scene:
                    frame, boxes = scene.step()
                    detector.boxes = boxes
                    time.sleep(min_dt)
                else:
                    if now - last_proc < min_dt:
                        cap.grab()                     # drop frames, keep stream fresh
                        continue
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        fails += 1
                        if fails > 30:
                            raise ConnectionError("camera stream lost")
                        time.sleep(0.1)
                        continue
                    fails = 0
                last_proc = time.time()

                dets = detector.detect(frame)
                for ev in engine.process(frame, dets, last_proc):
                    db.add_event(ev["direction"], ev["green_bag"], ev["track_id"])
                    self.log(f"[event] {ev['direction']} bag={ev['green_bag']}")

                vis = engine.visible()
                self._publish(frame, engine, vis)
                if last_proc - last_snap >= conf["snapshot_interval"]:
                    db.add_snapshot(db.get_inside(), len(vis))
                    last_snap = last_proc

                fps_n += 1
                if last_proc - fps_t0 >= 2:
                    self._status["fps"] = round(fps_n / (last_proc - fps_t0), 1)
                    fps_n, fps_t0 = 0, last_proc
                self._status["in_frame"] = len(vis)
        finally:
            if cap is not None:
                cap.release()
            self._status["connected"] = False

    def _publish(self, frame, engine, vis):
        h, w = frame.shape[:2]
        img = frame.copy()
        ax, ay, bx, by = [int(v) for v in engine.line_px(w, h)]
        cv2.line(img, (ax, ay), (bx, by), (0, 215, 255), 2)
        cv2.circle(img, (ax, ay), 6, (0, 215, 255), -1)           # A
        cv2.putText(img, "A", (ax + 8, ay - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
        for t in vis:
            x1, y1, x2, y2 = [int(v) for v in t.box[:4]]
            col = (60, 220, 60) if t.has_bag else (255, 160, 40)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"#{t.id}{' BAG' if t.has_bag else ''}", (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if w > 960:
            img = cv2.resize(img, (960, int(h * 960 / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._frame_id += 1
                self._cond.notify_all()
