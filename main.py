from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import time



def main_batch():
    model = YOLO("yolo11n.pt")
    model.export(format="engine")
    tensorrt_model = YOLO("yolo11n.engine")

    GST_PIPELINE = f"filesrc location={footage1} ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Cannot open connection!")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = tensorrt_model(frame, classes=[0])[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf.item())
            f.write(f"{frame_id},{cls},{x1},{y1},{x2},{y2},{score:.4f}\n")

    cap.release()
    out.release()