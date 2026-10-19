@echo off
REM Local Windows build (Python 3.10 - 3.12 recommended).  Output: dist\StoreCounter\
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
pip install -r requirements.txt pyinstaller || goto :err
if not exist models\yolov8n.onnx (
  pip install ultralytics onnx || goto :err
  python tools\export_model.py || goto :err
)
python -m unittest discover -s tests || goto :err
pyinstaller store_counter.spec --noconfirm || goto :err
xcopy /E /I /Y models dist\StoreCounter\models
copy /Y packaging\install_autostart.bat dist\StoreCounter\
echo.
echo DONE: dist\StoreCounter\StoreCounter.exe
exit /b 0
:err
echo BUILD FAILED
exit /b 1
