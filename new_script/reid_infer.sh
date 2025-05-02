#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT
set -x
CUR_DIR="$(pwd)"
CKPT="/kaggle/input/re-id-model/aic24.pkl"
CAMERAS=("c01" "c02")

DET_ROOT="${CUR_DIR}/custom_result/"
VID_ROOT="/kaggle/input/test-video/TestVideo/"
SAVE_ROOT="${CUR_DIR}/custom_result/"


for i in "${!CAMERAS[@]}"; do
  GPU_ID=$i

  CPU_START=$((i*2))
  CPU_END=$((i*2+1))

  export CUDA_VISIBLE_DEVICES=$GPU_ID

  taskset -c ${CPU_START}-${CPU_END} python custom/reid_infer.py \
    --cam_id $((i+1)) \
    --det_root $DET_ROOT \
    --vid_root $VID_ROOT \
    --save_root $SAVE_ROOT \
    --ckpt_path $CKPT \
    &

  echo "Started processing camera ${CAMERAS[$i]} on GPU $GPU_ID with CPU cores ${CPU_START}-${CPU_END}"

  sleep 2
done

wait

echo "All pose estimation tasks completed."