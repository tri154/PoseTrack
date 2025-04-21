from ultralytics import YOLO
import argparse
import os
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_path", type=str)
    parser.add_argument("--cam_id", type=str)

    args = parser.parse_args()

    root_folder = os.getcwd()
    output_path = os.path.join(root_folder, "custom_result", "cam" + args.cam_id + "_dets" + ".txt")

    engine_file = "yolo11l.engine"

    tensorrt_model = YOLO("yolo11l.engine")

    # GST_PIPELINE = f"filesrc location={args.cam_path} ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"
    GST_PIPELINE = f"filesrc location={args.cam_path} ! decodebin ! videoconvert ! appsink"
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
    #create file given ouput path
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        for det in dets:
            f.write(",".join(map(str, det)) + "\n")
    cap.release()
    print("Done!")

if __name__ == "__main__":
    main()
