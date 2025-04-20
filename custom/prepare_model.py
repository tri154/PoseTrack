
from ultralytics import YOLO
import os

def prepare_model():
    if os.path.exists("yolo11l.engine"):
        print("TensorRT engine already exists. Skipping export.")
        return
    print("Exporting YOLO model to TensorRT...")
    model = YOLO("yolo11l.pt")
    model.export(format="engine")
    print("Export complete.")

if __name__ == "__main__":
    prepare_model()
