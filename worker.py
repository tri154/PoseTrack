import os
import cv2
from ultralytics import YOLO
import numpy as np
from custom.pose_estimate import infer_one_image, get_pose_estimator
from custom.reid_infer import get_reid_model
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'track')))

from Tracker.PoseTracker import Detection_Sample, PoseTracker,TrackState
screen_width = 1920
screen_height = 1080
from multiprocessing import Process, Queue
from utils import logging
import traceback
def process_video(cam_id, cam_path, gpu_id, queue):
    log_file = f"progress_log_cam{cam_id}.txt"
    logging(log_file, "start processing")
    try:
        engine_file = "yolo11l.pt"
        #detection model
        tensorrt_model = YOLO(engine_file).to(f'cuda:{gpu_id}')
        logging(log_file, "Done detection")

        #pose estimation model
        pose_estimator = get_pose_estimator(f'cuda:{gpu_id}')
        logging(log_file, "Done pose")

        #reid model
        reid_model = get_reid_model(f'cuda:{gpu_id}')
        logging(log_file, "Done reid")

    except Exception as e:
        logging(log_file, str(e))
        logging(log_file, traceback.format_exc())

    logging(log_file, "loaded model")

    GST_PIPELINE = f"filesrc location={cam_path} ! decodebin ! videoconvert ! appsink"
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    frame_id = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        try:
            logging(log_file, f"Start {frame_id}")


            det_results = tensorrt_model(frame, classes=[0])[0]
            dets = list()
            for box in det_results.boxes:
                cls = int(box.cls.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                score = float(box.conf.item())
                dets.append([frame_id, cls, x1, y1, x2, y2, score])

            dets = np.array(dets)
            logging(log_file, "detected")


            #POSE estimate
            if len(dets) == 0:  # need to do smthing
                queue.put({
                    "is_end": False,
                    "camera_id": cam_id,
                    "frame_id": frame_id,
                    "detection_samples": [],
                })
                continue

            bboxes_s = dets[:, 2:7]  # x1y1x2y2s

            pose_result = infer_one_image(None, frame, bboxes_s, pose_estimator)
            pose_result = np.concatenate((np.ones((len(pose_result), 1)) * frame_id, pose_result.astype(np.float32)), axis=1)
            logging(log_file, "estimated pose")

            #reid
            bboxes_s = dets[:, 2:7]  # x1y1x2y2s
            x1 = bboxes_s[:, 0]
            y1 = bboxes_s[:, 1]
            x2 = bboxes_s[:, 2]
            y2 = bboxes_s[:, 3]

            x1 = np.maximum(0, x1)
            y1 = np.maximum(0, y1)
            x2 = np.minimum(screen_width, x2)
            y2 = np.minimum(screen_height, y2)

            bboxes_s[:, 0] = x1
            bboxes_s[:, 1] = y1
            bboxes_s[:, 2] = x2
            bboxes_s[:, 3] = y2
            with torch.no_grad():
                feat_sim = reid_model.process_frame_simplified(frame, bboxes_s[:, :-1])

            logging(log_file, "reid-ed")


            box_thred = 0.3
            detection_sample_sv = []
            for det, pose, reid in zip(dets, pose_result, feat_sim):  # doi voi moi detection trong do
                if det[-1] < box_thred or len(det) == 0:
                    continue  # loai bo confident thap
                new_sample = Detection_Sample(bbox=det[2:], keypoints_2d=pose[6:].reshape(17, 3), reid_feat=reid, cam_id=cam_id,
                                              frame_id=frame_id)
                detection_sample_sv.append(new_sample)
            queue.put({
                "is_end": False,
                "camera_id": cam_id,
                "frame_id": frame_id,
                "detection_samples": detection_sample_sv,
            })
            logging(log_file, f"Done {frame_id}")
        except Exception as e:
            logging(log_file, str(e))
            logging(log_file, traceback.format_exc())


    
    cap.release()
    queue.put({
        "is_end": True
    })

if __name__ == '__main__':
    q0 = Queue()
    process_video(0, "/kaggle/input/aic2024-sample/cam1-537/537_shorten.mp4", 0, q0)
