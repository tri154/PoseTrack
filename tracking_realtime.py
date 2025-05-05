from multiprocessing import Process, Queue
from worker import process_video
import os
from custom.prepare_model import prepare_model
from custom.tracking import get_pose_tracker
def main():

    engine_file = "yolo11l.engine"
    if not os.path.exists(engine_file):
        prepare_model()

    pose_tracker = get_pose_tracker()

    q0 = Queue()
    q1 = Queue()

    p0 = Process(target=process_video, args=(0, "video1.mp4", 0, q0))
    p1 = Process(target=process_video, args=(1, "video2.mp4", 1, q1))

    p0.start()
    p1.start()
    results = []
    while True:
        if not q0.empty() and not q1.empty():
            cam1_data = q0.get()
            cam2_data = q1.get()
            if cam1_data['is_end'] or cam2_data['is_end']:
                break
            frame_id = cam1_data["frame_id"]
            detection_sample_mv = [cam1_data["detection_samples"], cam2_data["detection_samples"]]
            pose_tracker.mv_update_wo_pred(detection_sample_mv, frame_id)
            frame_results = pose_tracker.output(frame_id)
            results += frame_results

    p0.join()
    p1.join()

if __name__ == "__main__":
    main()
