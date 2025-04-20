#!/usr/bin/env bash

# --- Configuration ---

# Get current working directory
CUR_DIR="$(pwd)"

# Camera 1
VIDEO_PATH_CAM1="/kaggle/input/test-video/TestVideo/output1.mp4"
DET_PATH_CAM1="${CUR_DIR}/custom_result/cam1_dets.txt"
OUTPUT_PATH_CAM1="${CUR_DIR}/custom_result/cam1_poses.txt"

# Camera 2
VIDEO_PATH_CAM2="/kaggle/input/test-video/TestVideo/output2.mp4"
DET_PATH_CAM2="${CUR_DIR}/custom_result/cam2_dets.txt"
OUTPUT_PATH_CAM2="${CUR_DIR}/custom_result/cam2_poses.txt"

# Define MMPose model paths
POSE_CONFIG="$/kaggle/input/ckpt-model/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"
POSE_CHECKPOINT="$/kaggle/input/ckpt-model/lup_moco_r101.pth"

# Define the Python script to run
PYTHON_SCRIPT="${CUR_DIR}/custom/pose_estimate.py"

# Optional: Define other parameters
BBOX_THR=0.3
KPT_THR=0.3

# --- Execution ---

function cleanup {
  echo "Cleaning up background processes..."
  PIDS=$(jobs -p)
  if [[ -n "$PIDS" ]]; then
    kill $PIDS 2>/dev/null
  fi
  echo "Cleanup done."
}

trap cleanup EXIT SIGINT SIGTERM

set -x

# Process Camera 1 on GPU 0, CPU cores 0-1
CUDA_VISIBLE_DEVICES=0 taskset -c 0-1 python ${PYTHON_SCRIPT} \
    "${POSE_CONFIG}" \
    "${POSE_CHECKPOINT}" \
    --video-path "${VIDEO_PATH_CAM1}" \
    --det-path "${DET_PATH_CAM1}" \
    --output-path "${OUTPUT_PATH_CAM1}" \
    --device cuda:0 \
    --bbox-thr ${BBOX_THR} \
    --kpt-thr ${KPT_THR} &
PID_CAM1=$!

# Process Camera 2 on GPU 1, CPU cores 2-3
CUDA_VISIBLE_DEVICES=1 taskset -c 2-3 python ${PYTHON_SCRIPT} \
    "${POSE_CONFIG}" \
    "${POSE_CHECKPOINT}" \
    --video-path "${VIDEO_PATH_CAM2}" \
    --det-path "${DET_PATH_CAM2}" \
    --output-path "${OUTPUT_PATH_CAM2}" \
    --device cuda:0 \
    --bbox-thr ${BBOX_THR} \
    --kpt-thr ${KPT_THR} &
PID_CAM2=$!

echo "Launched background processes:"
echo "  Camera 1 (PID ${PID_CAM1}) on GPU 0 / Cores 0-1"
echo "  Camera 2 (PID ${PID_CAM2}) on GPU 1 / Cores 2-3"
echo "Waiting for both processes to complete..."

wait ${PID_CAM1}
EXIT_STATUS_1=$?
wait ${PID_CAM2}
EXIT_STATUS_2=$?

set +x

if [ ${EXIT_STATUS_1} -ne 0 ]; then
    echo "Error: Camera 1 process (PID ${PID_CAM1}) failed with exit status ${EXIT_STATUS_1}."
fi
if [ ${EXIT_STATUS_2} -ne 0 ]; then
    echo "Error: Camera 2 process (PID ${PID_CAM2}) failed with exit status ${EXIT_STATUS_2}."
fi

if [ ${EXIT_STATUS_1} -eq 0 ] && [ ${EXIT_STATUS_2} -eq 0 ]; then
    echo "Both processes completed successfully."
else
    echo "One or both processes failed."
    exit 1
fi

echo "Script finished."
