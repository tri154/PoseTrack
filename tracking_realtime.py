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
import argparse
from util.camera import Camera



# VIDEO_1="/kaggle/input/aic2024-sample/cam1-537/537_shorten.mp4"
# VIDEO_2="/kaggle/input/aic2024-sample/cam2-543/543_shorten.mp4"
# # VIDEO_1="/kaggle/input/test-video-1/cam1.mp4"
# # VIDEO_2="/kaggle/input/test-video-1/cam2.mp4"
# SAVE_PATH="/kaggle/working/PoseTrack/custom_result/track_results.txt"

import datetime


def frame_id_to_timestamp(EPOCH_START, frame_id, fps=10):
    return (EPOCH_START + datetime.timedelta(seconds=frame_id / fps)).isoformat()

from utils import logging
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bootstrap_servers", type = str)
    parser.add_argument("--topic", type = str)
    parser.add_argument("--video1", type = str)
    parser.add_argument("--video2", type = str)
    parser.add_argument("--save_path", type = str, default = "/kaggle/working/PoseTrack/custom_result/track_results.txt")

    parser.add_argument("--cal1", type = str)
    parser.add_argument("--cal2", type = str)

    args = parser.parse_args()

    cals = [Camera(args.cal1, 1), Camera(args.cal2, 2)]

    VIDEO_1 = args.video1
    VIDEO_2 = args.video2
    SAVE_PATH = args.save_path
    topic = args.topic

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )


    EPOCH_START = None
    pose_tracker = get_pose_tracker(cals)

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
            try:
                # logging(log_file, "getting data")
                cam1_data = q0.get()
                cam2_data = q1.get()
                if cam1_data['is_end'] or cam2_data['is_end']:
                    break
                if EPOCH_START is None:
                    EPOCH_START = datetime.datetime.now(datetime.timezone.utc)
                frame_id = cam1_data["frame_id"]
                timestamp = frame_id_to_timestamp(EPOCH_START, frame_id)
                detection_sample_mv = [cam1_data["detection_samples"], cam2_data["detection_samples"]]
                pose_tracker.mv_update_wo_pred(detection_sample_mv, frame_id)
                frame_results = pose_tracker.output(frame_id)
                if len(frame_results) == 0:
                    continue
                frame_results = np.array(frame_results).squeeze(axis=1)
                # logging(log_file, frame_results.shape)


                frame_results_with_timestamp = np.hstack(
                    (frame_results[:, :-1], np.full((frame_results.shape[0], 1), timestamp)))

                # logging(log_file, [type(val) for val in frame_results_with_timestamp[0]])
                # with open(SAVE_PATH, 'a') as f:
                #     np.savetxt(f, frame_results_with_timestamp, fmt='%s %s %s %s %s %s %s %s %s %s')
                #     # for row in frame_results:
                #     #     np.savetxt(f, row[:, :-1], fmt='%d %d %d %d %d %d %d %f %f')
                #     f.write('\n')

                # producer.send('tracking', frame_results[:, :-1].tolist())


                producer.send(topic, frame_results_with_timestamp.tolist())
                logging(log_file, 'sent')
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
