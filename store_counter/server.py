"""Flask admin panel: live stats, reports (day/week/month), CSV export, settings."""
import csv
import datetime as dt
import hmac
import io
import os

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import jalali
from .config import web_dir


def create_app(cfg, db, pipeline):
    app = Flask(__name__, static_folder=None)

    @app.before_request
    def _auth():
        pw = cfg["admin_password"]
        if not pw:
            return None
        a = request.authorization
        if a and a.username == "admin" and hmac.compare_digest(a.password or "", pw):
            return None
        return Response("login required", 401, {"WWW-Authenticate": 'Basic realm="Store Counter"'})

    @app.get("/")
    def index():
        return send_from_directory(web_dir(), "index.html")

    @app.get("/api/live")
    def live():
        today = dt.date.today()
        return jsonify({
            "now": dt.datetime.now().strftime("%H:%M:%S"),
            "date": jalali.jalali_long(today),
            "inside": db.get_inside(),
            "today": db.summary(today.isoformat()),
            "status": pipeline.status(),
            "recent": db.recent_events(12),
        })

    def _date_arg():
        try:
            return dt.date.fromisoformat(request.args.get("date") or dt.date.today().isoformat())
        except ValueError:
            abort(400, "invalid date")

    def _period_arg():
        p = request.args.get("period", "day")
        if p not in ("day", "week", "month"):
            abort(400, "invalid period")
        return p

    @app.get("/api/report")
    def report():
        return jsonify(db.report(_period_arg(), _date_arg()))

    @app.get("/api/export.csv")
    def export_csv():
        r = db.report(_period_arg(), _date_arg())
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["بازه", "ورودی", "خروجی", "ورودی با کیف سبز", "خروجی با کیف سبز", "حداکثر حضور"])
        for x in r["rows"]:
            w.writerow([x["label"], x["in"], x["out"], x["bag_in"], x["bag_out"], x["peak"]])
        t = r["totals"]
        w.writerow(["جمع", t["in"], t["out"], t["bag_in"], t["bag_out"], t["peak"]])
        name = f"report_{r['period']}_{r['start']}.csv"
        return Response("﻿" + buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    @app.get("/video.mjpg")
    def video():
        def gen():
            last = -1
            while True:
                last, jpg = pipeline.wait_frame(last)
                if jpg:
                    yield (b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    @app.get("/api/config")
    def get_config():
        c = cfg.snapshot()
        c.pop("admin_password", None)
        return jsonify(c)

    @app.post("/api/config")
    def set_config():
        body = request.get_json(force=True, silent=True) or {}
        for k in ("admin_password", "data_dir", "host", "port", "model_path"):
            body.pop(k, None)               # not editable from the browser
        try:
            cfg.update(body)
        except (ValueError, TypeError, KeyError) as e:
            return jsonify({"error": str(e)}), 400
        pipeline.reload()
        return jsonify({"ok": True})

    @app.post("/api/reset_inside")
    def reset_inside():
        n = (request.get_json(silent=True) or {}).get("value", 0)
        db.set_inside(int(n))
        return jsonify({"ok": True, "inside": db.get_inside()})

    return app
