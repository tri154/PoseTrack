#!/usr/bin/env bash

# --- Configuration ---

# Define paths to your camera inputs and desired outputs
# Camera 1
VIDEO_PATH_CAM1="/kaggle/input/test-video/TestVideo/output1.mp4"
DET_PATH_CAM1="/custom_result/cam1_dets.txt"
OUTPUT_PATH_CAM1="/custom_result/cam1_poses.txt"

# Camera 2
VIDEO_PATH_CAM2="/kaggle/input/test-video/TestVideo/output2.mp4"
DET_PATH_CAM2="custom_result/cam2_dets.txt"
OUTPUT_PATH_CAM2="custom_result/cam2_poses.txt"

# Define MMPose model paths
# POSE_CONFIG="configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
# POSE_CHECKPOINT="../ckpt_weight/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"
# Use absolute paths or paths relative to the `mmpose` directory
POSE_CONFIG="/kaggle/input/ckpt-model/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth"
POSE_CHECKPOINT="/kaggle/input/ckpt-model/lup_moco_r101.pth"

# Define the Python script to run (the modified one)
PYTHON_SCRIPT="custom/pose_estimate.py" # Assuming it's in the mmpose/demo dir

# Optional: Define other parameters for the python script if needed
BBOX_THR=0.3
KPT_THR=0.3

# --- Execution ---

# Register a function to be called on exit to kill background jobs
function cleanup {
  echo "Cleaning up background processes..."
  # pkill -P $$ # Kill all direct child processes of this script
  # A potentially safer way if other non-python children might exist:
  # Find PIDs of the python scripts launched by this script and kill them
  PIDS=$(jobs -p)
  if [[ -n "$PIDS" ]]; then
    kill $PIDS 2>/dev/null
  fi
  echo "Cleanup done."
}

trap cleanup EXIT SIGINT SIGTERM # Trap exit, interrupt, terminate signals


# Enable command echo for debugging
set -x

# --- Launch Processes ---

# Process Camera 1 on GPU 0, CPU cores 0-1
CUDA_VISIBLE_DEVICES=0 taskset -c 0-1 python ${PYTHON_SCRIPT} \
    "${POSE_CONFIG}" \
    "${POSE_CHECKPOINT}" \
    --video-path "${VIDEO_PATH_CAM1}" \
    --det-path "${DET_PATH_CAM1}" \
    --output-path "${OUTPUT_PATH_CAM1}" \
    --device cuda:0 \
    --bbox-thr ${BBOX_THR} \
    --kpt-thr ${KPT_THR} & # Run in background
PID_CAM1=$! # Store PID of the first background process

# Process Camera 2 on GPU 1, CPU cores 2-3
CUDA_VISIBLE_DEVICES=1 taskset -c 2-3 python ${PYTHON_SCRIPT} \
    "${POSE_CONFIG}" \
    "${POSE_CHECKPOINT}" \
    --video-path "${VIDEO_PATH_CAM2}" \
    --det-path "${DET_PATH_CAM2}" \
    --output-path "${OUTPUT_PATH_CAM2}" \
    --device cuda:0 \
    --bbox-thr ${BBOX_THR} \
    --kpt-thr ${KPT_THR} & # Run in background
PID_CAM2=$! # Store PID of the second background process

# --- Wait for Completion ---
echo "Launched background processes:"
echo "  Camera 1 (PID ${PID_CAM1}) on GPU 0 / Cores 0-1"
echo "  Camera 2 (PID ${PID_CAM2}) on GPU 1 / Cores 2-3"
echo "Waiting for both processes to complete..."

# Wait for both specific PIDs (more robust than a general 'wait')
wait ${PID_CAM1}
EXIT_STATUS_1=$?
wait ${PID_CAM2}
EXIT_STATUS_2=$?

# Disable command echo
set +x

# --- Check Exit Status ---
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
    exit 1 # Exit script with error status
fi

# Cleanup will be called automatically on exit now by the trap

echo "Script finished."