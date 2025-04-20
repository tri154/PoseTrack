# Copyright (c) OpenMMLab. All rights reserved.
# Modified for single camera processing with specific paths
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
import sys # Added for explicit path check exits

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import time
from tqdm import tqdm

# Removed mmdet import block as detection is pre-computed

def infer_one_image(frame, bboxes_s, pose_estimator):
    # Filter bboxes by score if needed (using a fixed threshold or passed arg)
    # bbox_thr = 0.3 # Example threshold
    # bboxes_s = bboxes_s[bboxes_s[:, 4] > bbox_thr]
    # if len(bboxes_s) == 0:
    #    return np.zeros((0, 5 + 17*3), dtype=np.float32) # Adjust size based on keypoints

    pose_results = inference_topdown(pose_estimator, frame, bboxes_s[:, :4]) # Use only bbox coords
    records = []
    for i, result in enumerate(pose_results):
        # Handle cases where no keypoints are detected for a bbox if necessary
        if not result.pred_instances:
             # Create a placeholder or skip this detection
             num_keypoints = pose_estimator.cfg.data_cfg['num_keypoints'] # Get num_keypoints
             placeholder_kpts = np.zeros((num_keypoints, 3), dtype=np.float32) # x, y, score=0
             record = placeholder_kpts.flatten()
        else:
            keypoints = result.pred_instances.keypoints[0]
            scores = result.pred_instances.keypoint_scores.T # Shape (num_kpts, 1)
            record = np.concatenate((keypoints, scores), axis=1).flatten() # Shape (num_kpts * 3,)

        records.append(record)

    if not records: # If pose_results was empty or all detections failed pose
        # Adjust shape based on expected output cols (5 from bbox + num_kpts*3)
        num_keypoints = pose_estimator.cfg.data_cfg.get('num_keypoints', 17) # Default COCO
        return np.zeros((0, 5 + num_keypoints * 3), dtype=np.float32)

    records = np.array(records, dtype=np.float32)
    # Ensure bboxes_s and records have the same number of rows
    if records.shape[0] != bboxes_s.shape[0]:
        print(f"Warning: Mismatch between bbox count ({bboxes_s.shape[0]}) and pose results ({records.shape[0]})")
        # This case needs careful handling - maybe filter bboxes_s based on successful poses?
        # For now, assuming they match or pose_results might be shorter if some failed
        min_rows = min(records.shape[0], bboxes_s.shape[0])
        records = np.concatenate((bboxes_s[:min_rows, :], records[:min_rows, :]), axis=1)
    else:
        records = np.concatenate((bboxes_s, records), axis=1) # Combine bbox+score with kpt+score

    return records


def main():
    """Process a single video using pre-computed detections."""
    parser = ArgumentParser()
    # Removed det_config and det_checkpoint as detection is pre-computed
    parser.add_argument('pose_config', help='Config file for pose estimation model')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose estimation model')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--det-path', type=str, required=True, help='Path to the pre-computed detection file (.txt)')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the output pose results (.txt)')
    # Keep other relevant args
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference (e.g., cuda:0)')
    parser.add_argument(
        '--bbox-thr', # Keep if you want to filter detections inside python
        type=float,
        default=0.3,
        help='Bounding box score threshold (optional, applied after loading)')
    parser.add_argument(
        '--kpt-thr', # Relevant for visualization if that was added back
        type=float,
        default=0.3,
        help='Keypoint score threshold (for potential future use/filtering)')
    # Removed visualization/show arguments as the goal is saving predictions
    # Removed --input, --output-root, --start, --end etc.

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        sys.exit(1)
    if not os.path.exists(args.det_path):
        print(f"Error: Detection file not found at {args.det_path}")
        sys.exit(1)

    # --- Output Directory ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir: # Ensure output dir is not empty (for relative paths)
        os.makedirs(output_dir, exist_ok=True)

    # --- Initialize Pose Estimator ---
    try:
        pose_estimator = init_pose_estimator(
            args.pose_config,
            args.pose_checkpoint,
            device=args.device,
            cfg_options=dict(
                # Ensure test_cfg doesn't enable things not needed (like heatmaps unless required)
                model=dict(test_cfg=dict(output_heatmaps=False)))
        )
        # Get expected number of keypoints for output array sizing on empty frames/detections
        num_keypoints = pose_estimator.cfg.data_cfg.get('num_keypoints', 17) # Default to COCO's 17
        print(f"Pose model expects {num_keypoints} keypoints.")

    except Exception as e:
        print(f"Error initializing pose estimator: {e}")
        sys.exit(1)


    # --- Load Detections ---
    try:
        det_annot = np.loadtxt(args.det_path, delimiter=",")
        if det_annot.ndim == 1: # Handle case of single detection in file
            det_annot = det_annot.reshape(1, -1)
        print(f"Loaded {det_annot.shape[0]} detections from {args.det_path}")
        # Basic check on columns (at least frame_id, x1,y1,x2,y2,score = 6)
        # The original script expected 7 columns and used indices 2:7 (5 cols)
        if det_annot.shape[1] < 7:
             print(f"Warning: Detection file {args.det_path} has fewer than 7 columns.")
             # Adjust slicing if needed based on actual format
    except Exception as e:
        print(f"Error loading detection file {args.det_path}: {e}")
        sys.exit(1)

    # --- Process Video ---
    try:
        video = mmcv.VideoReader(args.video_path)
        print(f"Processing video: {args.video_path} ({len(video)} frames)")
    except Exception as e:
        print(f"Error opening video file {args.video_path}: {e}")
        sys.exit(1)

    all_results = []
    det_len = len(det_annot)
    processed_frames = 0

    for frame_id, frame in enumerate(tqdm(video, desc=f"Processing {os.path.basename(args.video_path)}")):
        if frame is None:
            print(f"Warning: Got empty frame at frame_id {frame_id}")
            continue

        # Get detections for the current frame
        # Ensure det_annot[:, 0] is integer-like for comparison if needed
        # frame_dets = det_annot[det_annot[:, 0].astype(int) == frame_id]
        # Robust check in case frame IDs in file are floats
        frame_dets = det_annot[np.isclose(det_annot[:, 0], frame_id)]

        if len(frame_dets) == 0:
            processed_frames += 1
            continue

        # Extract BBox [x1, y1, x2, y2, score]
        # Assuming format: frame, ?, x1, y1, x2, y2, score
        bboxes_s = frame_dets[:, 2:7]

        # Optional: Apply bbox threshold
        bboxes_s = bboxes_s[bboxes_s[:, 4] > args.bbox_thr]

        if len(bboxes_s) == 0:
             processed_frames += 1
             continue

        # --- Inference ---
        try:
            result = infer_one_image(frame, bboxes_s, pose_estimator) # Pass only frame, boxes, estimator
            if result.shape[0] > 0: # Only add if poses were found
                 # Add frame_id column
                 result_with_frame = np.concatenate((np.full((len(result), 1), frame_id, dtype=np.float32), result), axis=1)
                 all_results.append(result_with_frame)
        except Exception as e:
            print(f"Error during inference on frame {frame_id}: {e}")
            # Decide whether to continue or stop on error

        processed_frames += 1
        # Optional: Add a limit for debugging
        # if processed_frames >= 50:
        #     print_log("Debug limit reached.", logger='current')
        #     break

    print(f"Finished processing {processed_frames} frames.")

    # --- Save Results ---
    if not all_results:
        print("No results generated to save.")
        # Create an empty file or just don't save anything
        # open(args.output_path, 'w').close() # Example: create empty file
    else:
        try:
            all_results_np = np.concatenate(all_results, axis=0)
            # Save with space delimiter and sufficient precision
            np.savetxt(args.output_path, all_results_np, fmt='%.4f', delimiter=' ')
            print(f"Saved {all_results_np.shape[0]} pose results to {args.output_path}")
        except Exception as e:
            print(f"Error saving results to {args.output_path}: {e}")

    print("Processing finished for this camera.")


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    main()