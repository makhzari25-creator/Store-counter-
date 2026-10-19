"""Person detectors. All return a list of (x1, y1, x2, y2, conf) in frame pixels."""
import cv2
import numpy as np


class YoloOnnxDetector:
    """YOLOv8 (COCO, class 0 = person) exported to ONNX, run with OpenCV DNN.
    No PyTorch needed at runtime, so the exe stays small."""
    name = "yolo"

    def __init__(self, model_path: str, conf: float = 0.35, imgsz: int = 640, nms: float = 0.45):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf, self.imgsz, self.nms = conf, imgsz, nms

    def detect(self, frame):
        h, w = frame.shape[:2]
        s = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()
        return self.decode(out, w, h, s, px, py)

    def decode(self, out, w, h, s, px, py):
        out = np.squeeze(out)
        if out.ndim != 2:
            return []
        if out.shape[0] < out.shape[1]:      # (84, 8400) -> (8400, 84)
            out = out.T
        scores = out[:, 4]                   # class 0 (person) score
        keep = scores >= self.conf
        if not keep.any():
            return []
        o, sc = out[keep], scores[keep]
        boxes = []
        for cx, cy, bw, bh in o[:, :4]:
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
        idx = cv2.dnn.NMSBoxes(boxes, sc.tolist(), self.conf, self.nms)
        res = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = boxes[i]
            x1 = max(0, int((x - px) / s))
            y1 = max(0, int((y - py) / s))
            x2 = min(w - 1, int((x + bw - px) / s))
            y2 = min(h - 1, int((y + bh - py) / s))
            if x2 > x1 and y2 > y1:
                res.append((x1, y1, x2, y2, float(sc[i])))
        return res


class MotionDetector:
    """Fallback when no model file exists: moving blobs via background subtraction.
    Much less accurate than YOLO (groups of people merge, still people are missed)."""
    name = "motion"

    def __init__(self, min_area_frac: float = 0.004):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
        self.min_area_frac = min_area_frac
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2)) if w > 640 else frame
        k = w / small.shape[1]
        mask = self.bg.apply(small)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * small.shape[0] * small.shape[1]
        res = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            res.append((int(x * k), int(y * k), int((x + bw) * k), int((y + bh) * k), 0.5))
        return res


def build_detector(cfg, log=print):
    """Pick the detector according to config / available model file."""
    import os
    mode = cfg["detector"]
    model = cfg.resolve(cfg["model_path"])
    if mode in ("auto", "yolo") and os.path.exists(model):
        log(f"[detector] YOLO ONNX: {model}")
        return YoloOnnxDetector(model, cfg["conf"], cfg["imgsz"])
    if mode == "yolo":
        raise FileNotFoundError(f"model not found: {model}")
    log("[detector] model file not found -> using MOTION fallback (low accuracy)")
    return MotionDetector()
