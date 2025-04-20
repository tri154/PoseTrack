#!/usr/bin/env bash

# Clean up all child processes on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$
}
trap cleanup EXIT

set -x  # Print commands as they run

# Define your camera video inputs
video_1="/kaggle/input/test-video/TestVideo/output1.mp4"
video_2="/kaggle/input/test-video/TestVideo/output2.mp4"

# CPU binding configuration (optional)
cpu_cores_per_gpu=2 #Kaggle CPUs.

# Run detection for each video on separate GPU
# Camera 0 → GPU 0
export CUDA_VISIBLE_DEVICES=0
taskset -c 0-$((cpu_cores_per_gpu - 1)) python custom/detection.py --cam_path "$video_1" --cam-id 1 &

# Camera 1 → GPU 1
export CUDA_VISIBLE_DEVICES=1
taskset -c $((cpu_cores_per_gpu))-$(($cpu_cores_per_gpu * 2 - 1)) python custom/detection.py --cam_path "$video_2" --cam-id 2 &

# Wait for both processes to finish
wait
