"""Entry point: python main.py   (or StoreCounter.exe)"""
import logging
import os
import threading
import time
import webbrowser

from store_counter.config import Config
from store_counter.db import Database, DbSwitch
from store_counter.pipeline import Pipeline
from store_counter.server import create_app


def main():
    cfg = Config()
    data_dir = cfg.resolve(cfg["data_dir"])
    os.makedirs(data_dir, exist_ok=True)
    real_db = Database(os.path.join(data_dir, "counter.db"), cfg["reset_inside_daily"])
    demo_path = os.path.join(data_dir, "demo.db")
    for suffix in ("", "-wal", "-shm"):          # demo data is throw-away: start clean
        if os.path.exists(demo_path + suffix):
            os.remove(demo_path + suffix)
    db = DbSwitch(real_db, Database(demo_path, cfg["reset_inside_daily"]), cfg)

    pipe = Pipeline(cfg, db)
    pipe.start()
    app = create_app(cfg, db, pipe)

    host, port = cfg["host"], cfg["port"]
    url = f"http://127.0.0.1:{port}"
    print("=" * 56)
    print(f"  Store Counter  -  admin panel: {url}")
    print(f"  config: {cfg.path}")
    print("  Close this window to stop the program.")
    print("=" * 56)
    if cfg["open_browser"]:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        pipe.stop()
        time.sleep(0.3)
        db.close()


if __name__ == "__main__":
    main()
