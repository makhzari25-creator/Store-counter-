"""Synthetic scene used for demo mode and tests (no camera needed).

People (rectangles) walk across the counting line top->bottom ("in" with the default
line) or bottom->top ("out"). Some carry a GREEN bag, some a RED bag, some nothing.
Ground-truth counters let the tests verify the engine."""
import random

import cv2
import numpy as np

GREEN = (40, 175, 55)      # BGR
RED = (40, 40, 190)
BODY_COLORS = [(150, 90, 60), (90, 90, 150), (120, 120, 120), (60, 110, 160), (170, 150, 90)]


class SyntheticScene:
    def __init__(self, w=640, h=360, seed=1, spawn_every=14, max_people=None, speed=(5, 8)):
        self.w, self.h = w, h
        self.rnd = random.Random(seed)
        self.spawn_every = spawn_every
        self.max_people = max_people
        self.speed = speed
        self.people = []
        self.frame_no = 0
        self.spawned = 0
        self.truth = {"in": 0, "out": 0, "bag_in": 0, "bag_out": 0}
        bg = np.full((h, w, 3), 70, np.uint8)
        cv2.rectangle(bg, (0, 0), (w - 1, h - 1), (95, 95, 95), 6)
        self.bg = bg

    def done(self):
        return (self.max_people is not None and self.spawned >= self.max_people
                and not self.people)

    def _spawn(self):
        r = self.rnd
        going_in = r.random() < 0.6
        bag = r.choice(["green", "green", "red", "none"])
        bw, bh = 44, 80
        p = {"x": r.randint(60, self.w - 60), "bw": bw, "bh": bh,
             "vy": r.randint(*self.speed) * (1 if going_in else -1),
             "vx": r.uniform(-0.6, 0.6),
             "y": -bh if going_in else self.h + bh,
             "color": r.choice(BODY_COLORS), "bag": bag, "in": going_in}
        self.people.append(p)
        self.spawned += 1
        self.truth["in" if going_in else "out"] += 1
        if bag == "green":
            self.truth["bag_in" if going_in else "bag_out"] += 1

    def step(self):
        """Advance one frame -> (frame, ground-truth boxes)."""
        self.frame_no += 1
        if self.frame_no % self.spawn_every == 0 and \
                (self.max_people is None or self.spawned < self.max_people):
            self._spawn()
        frame = self.bg.copy()
        boxes = []
        for p in list(self.people):
            p["y"] += p["vy"]
            p["x"] += p["vx"]
            if p["y"] < -p["bh"] - 5 or p["y"] > self.h + p["bh"] + 5:
                self.people.remove(p)
                continue
            x1, y1 = int(p["x"] - p["bw"] / 2), int(p["y"] - p["bh"] / 2)
            x2, y2 = x1 + p["bw"], y1 + p["bh"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), p["color"], -1)
            cv2.circle(frame, ((x1 + x2) // 2, y1 - 2), 10, (190, 200, 215), -1)
            if p["bag"] != "none":
                col = GREEN if p["bag"] == "green" else RED
                cv2.rectangle(frame, (x1 - 14, y1 + 40), (x1 + 16, y1 + 72), col, -1)
            bx1, by1 = max(0, x1 - 14), max(0, y1 - 12)
            bx2, by2 = min(self.w - 1, x2), min(self.h - 1, y2)
            if bx2 - bx1 > 8 and by2 - by1 > 8:
                boxes.append((bx1, by1, bx2, by2, 0.9))
        frame = cv2.add(frame, np.random.randint(0, 6, frame.shape, dtype=np.uint8))
        return frame, boxes


class DemoDetector:
    """'Detector' that returns the scene's ground-truth boxes."""
    name = "demo"

    def __init__(self):
        self.boxes = []

    def detect(self, frame):
        return self.boxes
