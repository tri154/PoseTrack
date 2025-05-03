# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import time
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def infer_one_image(args, frame, bboxes_s, pose_estimator):
    pose_results = inference_topdown(pose_estimator, frame, bboxes_s[:, :4])
    records = []
    for i, result in enumerate(pose_results):
        keypoints = result.pred_instances.keypoints[0]
        scores = result.pred_instances.keypoint_scores.T
        record = (np.concatenate((keypoints, scores), axis=1)).flatten()
        records.append(record)
    records = np.array(records)
    records = np.concatenate((bboxes_s, records), axis=1)
    return records


def main():
    """Process pose estimation for a specific scene with specific cameras."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--camera_id', type=str, required=True,
        help='Camera id.')
    parser.add_argument(
        '--det-root', type=str, required=True,
        help='Root directory of detection results')
    parser.add_argument(
        '--vid-root', type=str, required=True,
        help='Root directory of video files')
    parser.add_argument(
        '--save-root', type=str, required=True,
        help='Root directory for saving pose estimation results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))))

    cam_id = args.camera_id
    det_root = args.det_root
    vid_root = args.vid_root
    save_root = args.save_root

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print(f"Processing camera: {cam_id}")
    det_path = os.path.join(det_root, "cam" + cam_id + "_dets.txt")
    # vid_path = os.path.join(vid_root, "output" + cam_id + ".mp4")
    vid_path = ""
    if cam_id == 1:
        vid_path = os.path.join(vid_root, 'cam1-537', '537.mp4')
    elif cam_id == 2:
        vid_path = os.path.join(vid_root, 'cam2-543', '543.mp4')

    save_path = os.path.join(save_root, "cam" + cam_id + "_poses" + ".txt")

    # Skip if result already exists
    if os.path.exists(save_path):
        print(f"Results for {cam_id} already exist, skipping...")

    # Check if detection and video files exist
    if not os.path.exists(det_path):
        print(f"Detection file not found: {det_path}")

    if not os.path.exists(vid_path):
        print(f"Video file not found: {vid_path}")

    det_annot = np.loadtxt(det_path, delimiter=",")

    # Load video
    video = mmcv.VideoReader(vid_path)
    all_results = []

    # Process each frame
    for frame_id, frame in enumerate(tqdm(video)):
        dets = det_annot[det_annot[:, 0] == frame_id]
        bboxes_s = dets[:, 2:7]  # x1y1x2y2s

        if len(bboxes_s) == 0:
            continue

        result = infer_one_image(args, frame, bboxes_s, pose_estimator)
        result = np.concatenate((np.ones((len(result), 1)) * frame_id, result.astype(np.float32)), axis=1)
        all_results.append(result)

    # Save results
    if all_results:
        all_results = np.concatenate(all_results)
        np.savetxt(save_path, all_results)
        print(f"Results saved to: {save_path}")
    else:
        print(f"No results found for {cam_id}")


if __name__ == '__main__':
    main()