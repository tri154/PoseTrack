#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

set -x
CUR_DIR="$(pwd)"
CONFIG="mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
CKPT="/kaggle/input/ckpt-model/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"
# Specify scene and cameras
CAMERAS=("c01" "c02")  # Change these to your camera names

# Specify paths
DET_ROOT="${CUR_DIR}/custom_result/"  # Path to detection results
VID_ROOT="/kaggle/input/test-video/TestVideo/"  # Path to video files
SAVE_ROOT="${CUR_DIR}/custom_result/"  # Path to save pose estimation results

# Process each camera on a separate GPU
for i in "${!CAMERAS[@]}"; do
  # Assign GPU ID (alternating between 0 and 1)
  GPU_ID=$i  # This will be 0 for first camera, 1 for second camera

  # Assign CPU cores - 2 cores per task
  # First task: cores 0-1, Second task: cores 2-3
  CPU_START=$((i*2))
  CPU_END=$((i*2+1))  # Each task gets 2 cores

  # Set GPU device for this process
  export CUDA_VISIBLE_DEVICES=$GPU_ID

  # Run pose estimation for this camera in the background with taskset
  taskset -c ${CPU_START}-${CPU_END} python custom/pose_estimate.py \
    $CONFIG \
    $CKPT \
    --camera_id $((i+1)) \
    --det-root $DET_ROOT \
    --vid-root $VID_ROOT \
    --save-root $SAVE_ROOT \
    --device cuda:0 \
    &

  echo "Started processing camera ${CAMERAS[$i]} on GPU $GPU_ID with CPU cores ${CPU_START}-${CPU_END}"

  # Sleep briefly to avoid initialization conflicts
  sleep 2
done

# Wait for all background processes to complete
wait

echo "All pose estimation tasks completed."