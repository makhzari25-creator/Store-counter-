# PyInstaller spec -> dist/StoreCounter/StoreCounter.exe  (one-folder build)
# Build on Windows:  pyinstaller store_counter.spec --noconfirm
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ["main.py"],
    pathex=["."],
    datas=[("store_counter/web", "store_counter/web")],
    hiddenimports=collect_submodules("flask") + ["sqlite3"],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "torch", "ultralytics"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="StoreCounter",
          console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="StoreCounter", upx=False)
