from multiprocessing import Process, Queue
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from worker import process_video
import os
from custom.prepare_model import prepare_model
from custom.tracking import get_pose_tracker
import numpy as np
from kafka import KafkaProducer
import json
import traceback
producer = KafkaProducer(
    bootstrap_servers='42.118.0.103:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


VIDEO_1="/kaggle/input/aic2024-sample/cam1-537/537_shorten.mp4"
VIDEO_2="/kaggle/input/aic2024-sample/cam2-543/543_shorten.mp4"
SAVE_PATH="/kaggle/working/PoseTrack/custom_result/track_results.txt"

import datetime


def frame_id_to_timestamp(EPOCH_START, frame_id, fps=10):
    return (EPOCH_START + datetime.timedelta(seconds=frame_id / fps)).isoformat()

from utils import logging
def main():
    # engine_file = "yolo11l.engine"
    # if not os.path.exists(engine_file):
    #     prepare_model()
    EPOCH_START = None
    pose_tracker = get_pose_tracker()

    q0 = Queue()
    q1 = Queue()

    p0 = Process(target=process_video, args=(0, VIDEO_1, 0, q0))
    p1 = Process(target=process_video, args=(1, VIDEO_2, 1, q1))

    p0.start()
    p1.start()
    results = []

    log_file = "progress_log.txt"

    import time
    start_time = time.time()
    logging(log_file, "start tracking")
    while True:
        if not q0.empty() and not q1.empty():
            logging(log_file, "getting data")
            cam1_data = q0.get()
            cam2_data = q1.get()
            if cam1_data['is_end'] or cam2_data['is_end']:
                break
            # try:
            if EPOCH_START is None:
                EPOCH_START = datetime.datetime.now(datetime.timezone.utc)
            logging(log_file, "getting data Done")
            frame_id = cam1_data["frame_id"]
            timestamp = frame_id_to_timestamp(EPOCH_START, frame_id)
            detection_sample_mv = [cam1_data["detection_samples"], cam2_data["detection_samples"]]
            pose_tracker.mv_update_wo_pred(detection_sample_mv, frame_id)
            frame_results = pose_tracker.output(frame_id)
            frame_results = np.array(frame_results)
            # logging(log_file, type(frame_results))
            # except Exception as e:
            #     logging(log_file, str(e))
            try:
                # with open(SAVE_PATH, 'a') as f:
                #     np.savetxt(f, frame_results[:, :-1], fmt='%d %d %d %d %d %d %d %f %f')
                #     # for row in frame_results:
                #     #     np.savetxt(f, row[:, :-1], fmt='%d %d %d %d %d %d %d %f %f')
                #     f.write('\n')
                frame_results_with_timestamp = np.hstack(
                    (frame_results[:, :-1], np.full((frame_results.shape[0], 1), timestamp)))
                # producer.send('tracking', frame_results[:, :-1].tolist())
                logging(log_file, 'sending')
                producer.send('tracking', frame_results_with_timestamp.tolist())
                print("Sent")
            except Exception as e:
                logging(log_file, str(traceback.format_exc()))
            # results += frame_results
            logging(log_file, f"Done {frame_id}")
    p0.join()
    p1.join()

    producer.send('tracking', "Done")
    producer.flush()
    producer.close()

    end_time = time.time()
    fps = frame_id / (end_time - start_time)
    logging(log_file, f"FPS: {fps}")
    # results = np.concatenate(results,axis=0)
    # sort_idx = np.lexsort((results[:,2],results[:,0]))
    # results = np.ascontiguousarray(results[sort_idx])
    # np.savetxt(SAVE_PATH, results)

if __name__ == "__main__":
    main()
