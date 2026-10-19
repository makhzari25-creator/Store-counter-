"""Tracking + line crossing + green-bag detection. Pure logic: no camera, no DB.

engine.process(frame, detections, now) -> list of events
    event = {"direction": "in"|"out", "green_bag": bool, "track_id": int}
"""
import math
from collections import deque

import cv2
import numpy as np


class Track:
    def __init__(self, tid, box, now):
        self.id = tid
        self.box = box
        self.hits = 1
        self.missed = 0
        self.side = 0                      # -1 / +1 once far enough from the line
        self.green = deque(maxlen=40)      # per-frame "green found" flags
        self.last_ratio = 0.0
        self.has_bag = False

    @property
    def center(self):
        x1, y1, x2, y2 = self.box[:4]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def green_ratio(frame, box, g) -> float:
    """Share of 'green' pixels inside the (slightly shrunk) person box."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    mx, my = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
    crop = frame[max(0, y1 + my):max(0, y2 - my), max(0, x1 + mx):max(0, x2 - mx)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (g["h_min"], g["s_min"], g["v_min"]), (g["h_max"], 255, 255))
    return float(cv2.countNonZero(mask)) / mask.size


class CountingEngine:
    def __init__(self, cfg: dict, max_missed: int = 8):
        self.cfg = cfg                      # plain dict (a snapshot of the config)
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1
        self.frame_size = None

    # -- geometry ------------------------------------------------------------
    def line_px(self, w, h):
        l = self.cfg["line"]
        return l["x1"] * w, l["y1"] * h, l["x2"] * w, l["y2"] * h

    @staticmethod
    def _side_dist(p, line):
        ax, ay, bx, by = line
        length = math.hypot(bx - ax, by - ay) or 1.0
        cross = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
        t = ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / (length * length)
        return cross / length, t

    # -- main ----------------------------------------------------------------
    def process(self, frame, detections, now=None):
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        line = self.line_px(w, h)
        margin = max(4.0, 0.012 * h)
        max_dist = 0.15 * w

        tracks = list(self.tracks.values())
        pairs = []
        for ti, t in enumerate(tracks):
            cx, cy = t.center
            for di, d in enumerate(detections):
                dist = math.hypot(cx - (d[0] + d[2]) / 2.0, cy - (d[1] + d[3]) / 2.0)
                if dist <= max_dist:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        matches = []
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((tracks[ti], detections[di]))

        for ti, t in enumerate(tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                t = Track(self.next_id, d, now)
                self.next_id += 1
                self.tracks[t.id] = t
                matches.append((t, d))
        for tid in [i for i, t in self.tracks.items() if t.missed > self.max_missed]:
            del self.tracks[tid]

        events = []
        g = self.cfg["green"]
        for t, d in matches:
            if t.box is not d:
                t.hits += 1
            t.box = d
            t.missed = 0
            t.last_ratio = green_ratio(frame, d, g)
            t.green.append(t.last_ratio >= g["min_ratio"])
            n_green = sum(t.green)
            t.has_bag = n_green >= 2 and n_green / len(t.green) >= g["min_frame_fraction"]

            dist, tpos = self._side_dist(t.center, line)
            if abs(dist) < margin:
                continue                       # still on the line: wait for a clear side
            side = 1 if dist > 0 else -1
            if t.side and side != t.side and -0.05 <= tpos <= 1.05 \
                    and t.hits >= self.cfg["min_track_hits"]:
                going_in = side > 0
                if self.cfg["invert_direction"]:
                    going_in = not going_in
                events.append({"direction": "in" if going_in else "out",
                               "green_bag": t.has_bag, "track_id": t.id})
            t.side = side
        return events

    def visible(self):
        return [t for t in self.tracks.values() if t.missed == 0]
