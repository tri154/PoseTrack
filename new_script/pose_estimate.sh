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
CAMERAS=("c01" "c02")


DET_ROOT="${CUR_DIR}/custom_result/"
#VID_ROOT="/kaggle/input/test-video/TestVideo/"
VID_ROOT='/kaggle/input/aic2024-sample/'
SAVE_ROOT="${CUR_DIR}/custom_result/"

for i in "${!CAMERAS[@]}"; do
  GPU_ID=$i

  CPU_START=$((i*2))
  CPU_END=$((i*2+1))

  export CUDA_VISIBLE_DEVICES=$GPU_ID

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

  sleep 2
done

wait

echo "All pose estimation tasks completed."