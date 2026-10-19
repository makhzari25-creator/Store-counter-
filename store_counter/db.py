"""SQLite storage: events, occupancy snapshots, reports (day / week / month)."""
import datetime as dt
import sqlite3
import threading

from . import jalali

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  ts        TEXT    NOT NULL,              -- local time 'YYYY-MM-DD HH:MM:SS'
  day       TEXT    NOT NULL,              -- 'YYYY-MM-DD'
  hour      INTEGER NOT NULL,
  direction TEXT    NOT NULL CHECK(direction IN ('in','out')),
  green_bag INTEGER NOT NULL DEFAULT 0,
  track_id  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
CREATE TABLE IF NOT EXISTS snapshots(
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  ts       TEXT    NOT NULL,
  day      TEXT    NOT NULL,
  hour     INTEGER NOT NULL,
  inside   INTEGER NOT NULL,               -- estimated people inside the store
  in_frame INTEGER NOT NULL DEFAULT 0      -- people visible to the camera now
);
CREATE INDEX IF NOT EXISTS idx_snap_day ON snapshots(day);
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
"""


def _today() -> str:
    return dt.date.today().isoformat()


class Database:
    def __init__(self, path: str, reset_daily: bool = True):
        self.reset_daily = reset_daily
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.close()

    # ---- state / occupancy -------------------------------------------------
    def _get(self, key, default=None):
        r = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def _set(self, key, value):
        self.conn.execute("INSERT INTO state(key,value) VALUES(?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, str(value)))

    def _rollover(self, day: str):
        if self.reset_daily and self._get("inside_day") != day:
            self._set("inside", 0)
            self._set("inside_day", day)

    def get_inside(self) -> int:
        with self.lock:
            self._rollover(_today())
            self.conn.commit()
            return int(self._get("inside", 0))

    def set_inside(self, n: int):
        with self.lock:
            self._set("inside", max(0, int(n)))
            self._set("inside_day", _today())
            self.conn.commit()

    # ---- writes ------------------------------------------------------------
    def add_event(self, direction: str, green_bag: bool, track_id=None, when=None):
        when = when or dt.datetime.now()
        day = when.date().isoformat()
        with self.lock:
            self._rollover(day)
            inside = int(self._get("inside", 0)) + (1 if direction == "in" else -1)
            self._set("inside", max(0, inside))
            self.conn.execute(
                "INSERT INTO events(ts,day,hour,direction,green_bag,track_id) VALUES(?,?,?,?,?,?)",
                (when.strftime("%Y-%m-%d %H:%M:%S"), day, when.hour, direction,
                 1 if green_bag else 0, track_id))
            self.conn.commit()

    def add_snapshot(self, inside: int, in_frame: int = 0, when=None):
        when = when or dt.datetime.now()
        with self.lock:
            self.conn.execute(
                "INSERT INTO snapshots(ts,day,hour,inside,in_frame) VALUES(?,?,?,?,?)",
                (when.strftime("%Y-%m-%d %H:%M:%S"), when.date().isoformat(), when.hour,
                 inside, in_frame))
            self.conn.commit()

    # ---- reads -------------------------------------------------------------
    def summary(self, start: str, end: str = None) -> dict:
        end = end or start
        with self.lock:
            rows = self.conn.execute(
                "SELECT direction, green_bag, COUNT(*) c FROM events "
                "WHERE day BETWEEN ? AND ? GROUP BY direction, green_bag", (start, end)).fetchall()
        t = {"in": 0, "out": 0, "bag_in": 0, "bag_out": 0}
        for r in rows:
            t[r["direction"]] += r["c"]
            if r["green_bag"]:
                t["bag_" + r["direction"]] += r["c"]
        return t

    def recent_events(self, limit: int = 15):
        with self.lock:
            rows = self.conn.execute(
                "SELECT ts, direction, green_bag FROM events ORDER BY id DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def report(self, period: str, date: dt.date) -> dict:
        if period == "day":
            start = end = date
            title = jalali.jalali_long(date)
        elif period == "week":
            start, end = jalali.week_range(date)
            title = f"هفته {jalali.jalali_str(start)} تا {jalali.jalali_str(end)}"
        elif period == "month":
            start, end, title = jalali.jalali_month_range(date)
        else:
            raise ValueError("period must be day, week or month")
        s, e = start.isoformat(), end.isoformat()

        with self.lock:
            ev = self.conn.execute(
                "SELECT day, hour, direction, green_bag, COUNT(*) c FROM events "
                "WHERE day BETWEEN ? AND ? GROUP BY day, hour, direction, green_bag",
                (s, e)).fetchall()
            pk = self.conn.execute(
                "SELECT day, hour, MAX(inside) m FROM snapshots "
                "WHERE day BETWEEN ? AND ? GROUP BY day, hour", (s, e)).fetchall()

        if period == "day":
            keys = [f"{h:02d}" for h in range(24)]
            labels = [f"{h:02d}:00" for h in range(24)]
            keyof = lambda r: f"{r['hour']:02d}"
        else:
            n = (end - start).days + 1
            days = [start + dt.timedelta(days=i) for i in range(n)]
            keys = [d.isoformat() for d in days]
            labels = [(f"{jalali.WEEKDAYS[d.weekday()]} {jalali.jalali_str(d)[5:]}"
                       if period == "week" else jalali.jalali_str(d)[5:]) for d in days]
            keyof = lambda r: r["day"]

        buckets = {k: {"key": k, "label": l, "in": 0, "out": 0,
                       "bag_in": 0, "bag_out": 0, "peak": 0}
                   for k, l in zip(keys, labels)}
        for r in ev:
            b = buckets[keyof(r)]
            b[r["direction"]] += r["c"]
            if r["green_bag"]:
                b["bag_" + r["direction"]] += r["c"]
        for r in pk:
            k = keyof(r)
            buckets[k]["peak"] = max(buckets[k]["peak"], r["m"])

        rows = [buckets[k] for k in keys]
        totals = {f: sum(b[f] for b in rows) for f in ("in", "out", "bag_in", "bag_out")}
        totals["peak"] = max((b["peak"] for b in rows), default=0)
        return {"period": period, "start": s, "end": e, "title": title,
                "rows": rows, "totals": totals}
