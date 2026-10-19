import datetime as dt
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from store_counter import jalali                       # noqa: E402
from store_counter.config import DEFAULTS, Config      # noqa: E402
from store_counter.db import Database                  # noqa: E402
from store_counter.demo import DemoDetector, SyntheticScene   # noqa: E402
from store_counter.detectors import YoloOnnxDetector   # noqa: E402
from store_counter.engine import CountingEngine        # noqa: E402


def run_scene(seed, n=40, invert=False):
    scene = SyntheticScene(seed=seed, max_people=n)
    det = DemoDetector()
    cfg = {**DEFAULTS, "invert_direction": invert}
    eng = CountingEngine(cfg)
    got = {"in": 0, "out": 0, "bag_in": 0, "bag_out": 0}
    guard = 0
    while not scene.done() and guard < 20000:
        guard += 1
        frame, det.boxes = scene.step()
        for ev in eng.process(frame, det.detect(frame)):
            got[ev["direction"]] += 1
            if ev["green_bag"]:
                got["bag_" + ev["direction"]] += 1
    return got, scene.truth


class JalaliTests(unittest.TestCase):
    def test_known_dates(self):
        self.assertEqual(jalali.gregorian_to_jalali(2024, 3, 20), (1403, 1, 1))
        self.assertEqual(jalali.gregorian_to_jalali(2026, 10, 19), (1405, 7, 27))

    def test_round_trip(self):
        d = dt.date(2020, 1, 1)
        for i in range(0, 3000, 7):
            x = d + dt.timedelta(days=i)
            self.assertEqual(dt.date(*jalali.jalali_to_gregorian(*jalali.to_jalali(x))), x)

    def test_week_starts_saturday(self):
        s, e = jalali.week_range(dt.date(2026, 10, 19))       # Monday
        self.assertEqual(s.weekday(), 5)
        self.assertEqual((e - s).days, 6)

    def test_month_range(self):
        s, e, _ = jalali.jalali_month_range(dt.date(2026, 10, 19))   # Mehr 1405
        self.assertEqual(jalali.to_jalali(s), (1405, 7, 1))
        self.assertEqual(jalali.to_jalali(e), (1405, 7, 30))


class EngineTests(unittest.TestCase):
    def test_counts_match_ground_truth(self):
        for seed in (1, 2, 3):
            got, truth = run_scene(seed)
            self.assertEqual(got, truth, f"seed {seed}: got {got} truth {truth}")

    def test_invert_swaps_direction(self):
        got, truth = run_scene(5, n=20, invert=True)
        self.assertEqual(got["in"], truth["out"])
        self.assertEqual(got["out"], truth["in"])


class DbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmp, "t.db"))

    def tearDown(self):
        self.db.close()

    def test_events_occupancy_and_reports(self):
        d = dt.date.today()
        t = dt.datetime.combine(d, dt.time(10, 5))
        for _ in range(5):
            self.db.add_event("in", False, when=t)
        self.db.add_event("in", True, when=t)
        self.db.add_event("out", True, when=t.replace(hour=11))
        self.db.add_event("out", False, when=t.replace(hour=11))
        self.assertEqual(self.db.get_inside(), 4)
        self.db.add_snapshot(6, 2, when=t)

        day = self.db.report("day", d)
        self.assertEqual(day["totals"], {"in": 6, "out": 2, "bag_in": 1, "bag_out": 1, "peak": 6})
        self.assertEqual(day["rows"][10]["in"], 6)
        self.assertEqual(day["rows"][11]["out"], 2)
        self.assertEqual(len(day["rows"]), 24)
        self.assertEqual(len(self.db.report("week", d)["rows"]), 7)
        month = self.db.report("month", d)
        self.assertGreaterEqual(len(month["rows"]), 29)
        self.assertEqual(month["totals"]["in"], 6)

    def test_occupancy_never_negative_and_resets_daily(self):
        self.db.add_event("out", False)
        self.assertEqual(self.db.get_inside(), 0)
        self.db.add_event("in", False)
        self.db._set("inside_day", "2000-01-01")             # pretend it is a new day
        self.assertEqual(self.db.get_inside(), 0)


class ConfigTests(unittest.TestCase):
    def test_update_validates(self):
        c = Config(os.path.join(tempfile.mkdtemp(), "c.json"))
        c.update({"conf": 5, "line": {"y1": 0.3, "y2": 0.3}})
        self.assertEqual(c["conf"], 0.95)
        with self.assertRaises(ValueError):
            c.update({"line": {"x1": 0.1, "y1": 0.1, "x2": 0.1, "y2": 0.1}})


class YoloDecodeTests(unittest.TestCase):
    def test_decode_letterbox_and_nms(self):
        det = YoloOnnxDetector.__new__(YoloOnnxDetector)      # skip loading a model
        det.conf, det.imgsz, det.nms = 0.35, 640, 0.45
        out = np.zeros((1, 84, 100), np.float32)              # 100 candidates (like 84 x 8400)
        out[0, :4, 0] = [320, 320, 100, 200]; out[0, 4, 0] = 0.9     # person
        out[0, :4, 1] = [322, 322, 100, 200]; out[0, 4, 1] = 0.8     # duplicate -> NMS
        out[0, :4, 2] = [50, 50, 40, 40];     out[0, 4, 2] = 0.1     # below conf
        # frame 1280x720 -> scale 0.5, letterbox pad y = (640-360)/2 = 140
        res = det.decode(out, 1280, 720, 0.5, 0, 140)
        self.assertEqual(len(res), 1)
        x1, y1, x2, y2, c = res[0]
        self.assertAlmostEqual(x1, (270) / 0.5, delta=2)
        self.assertAlmostEqual(x2, (370) / 0.5, delta=2)
        self.assertAlmostEqual(y1, (220 - 140) / 0.5, delta=2)
        self.assertAlmostEqual(y2, (420 - 140) / 0.5, delta=2)


if __name__ == "__main__":
    unittest.main()
