"""Download YOLOv8n (COCO) and export it to models/yolov8n.onnx for OpenCV DNN.

Needs (only for this one-off step):  pip install ultralytics onnx
Run from the repo root:               python tools/export_model.py
"""
import os
import shutil

from ultralytics import YOLO

os.makedirs("models", exist_ok=True)
out = YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12, simplify=True)
shutil.move(out, os.path.join("models", "yolov8n.onnx"))
print("OK -> models/yolov8n.onnx")
