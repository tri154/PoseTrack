#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

set -x
CUR_DIR="$(pwd)"
RESULT_ROOT="${CUR_DIR}/custom_result/"
CAL_ROOT="/kaggle/input/aic2024-sample"
SAVE_ROOT="${CUR_DIR}/custom_result/"


python custom/tracking.py --result_root $RESULT_ROOT --calibrate_root $CAL_ROOT --save_root $SAVE_ROOT > result/track_log/$i.txt

wait