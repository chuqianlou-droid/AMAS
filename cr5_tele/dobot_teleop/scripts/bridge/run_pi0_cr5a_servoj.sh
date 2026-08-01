#!/usr/bin/env bash
set -euo pipefail

cd /home/jiaotan/AMAS/cr5_tele/dobot_teleop
source ~/miniconda3/etc/profile.d/conda.sh
conda activate teleop_dobot

export PYTHONPATH=/home/jiaotan/AMAS/cr5_tele/openpi/packages/openpi-client/src:${PYTHONPATH:-}
export CR5A_D415_SERIAL=${CR5A_D415_SERIAL:-841612070371}
export CR5A_D435_SERIAL=${CR5A_D435_SERIAL:-801312070525}

INSTRUCTION="${1:-pick the object}"

python3 scripts/bridge/pi0_cr5a_bridge.py \
  --robot-ip 192.168.5.1 \
  --policy-host 127.0.0.1 --policy-port 8000 \
  --observation-provider scripts/bridge/cr5a_observation_provider.py:make_observation \
  --instruction "$INSTRUCTION" \
  --action-format cartesian_delta_mm_deg \
  --delta-frame base \
  --servo-mode joint \
  --command-rate 25 \
  --servo-t 0.067 \
  --servo-aheadtime 100 \
  --servo-gain 200 \
  --target-lowpass-alpha 0.25 \
  --max-linear-speed-mm-s 25 \
  --max-angular-speed-deg-s 8 \
  --max-joint-speed-deg-s 30 \
  --max-total-translation-mm 250 \
  --max-total-rotation-deg 80 \
  --no-use-gripper-center-pose \
  --enable-gripper \
  --gripper-force 80 \
  --gripper-speed 80 \
  --gripper-trigger-threshold 0.7 \
  --gripper-open-threshold 0.25 \
  --gripper-min-command-interval-s 1.0 \
  --gripper-close-delay-s 0.7 \
  --gripper-close-max-lag-mm 15 \
  --clear-error \
  --enable-robot \
  --execute \
  --log-targets
