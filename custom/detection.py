from ultralytics import YOLO
import argparse
import os
import cv2


def prepare_model(device=0):
    print(f"Exporting YOLO model to TensorRT on GPU:{device}")
    model = YOLO(f"yolo11l.pt")
    exported_file = model.export(format="engine", device=device)
    new_file = f"yolo11l_{device}.engine"
    os.rename(exported_file, new_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_path", type=str)
    parser.add_argument("--cam_id", type=str)
    parser.add_argument("--device", type=int, default=0, help="GPU device to use for export/inference")

    args = parser.parse_args()
    output_path = args.cam_id + ".txt"

    prepare_model(args.device)

    tensorrt_model = YOLO(f"yolo11l_{args.device}.engine")

    GST_PIPELINE = f"filesrc location={args.cam_path} ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Cannot open connection!")
    dets = list()
    frame_id = -1
    while cap.isOpened():
        frame_id += 1
        ret, frame = cap.read()

        if not ret:
            break

        results = tensorrt_model(frame, classes=[0])[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf.item())
            dets.append([frame_id, cls, x1, y1, x2, y2, score])

    #write dets to output_path
    with open(output_path, "w") as f:
        for det in dets:
            f.write(",".join(map(str, det)) + "\n")
    cap.release()
    print("Done!")

if __name__ == "__main__":
    main()