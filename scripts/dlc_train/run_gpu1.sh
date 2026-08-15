#!/usr/bin/env bash
# DLC training, half of the models, GPU 1, cores 12-23.
#
#   dekr_w48    65.33M   batch 16     the most expensive model in this half
#   dekr_w18     9.56M   batch 32
#   resnet_101  42.50M   batch 24
#   hrnet_w32   29.31M   batch 24
#   dlcrnet_stride16_ms5  batch 16
#
# Paired with run_gpu0.sh, which takes the other five on GPU 0 and cores
# 0-11. The two halves are balanced by measured cost, not by model count: one
# W48 goes to each script rather than both to one, and each half carries one
# DLCRNet.
#
# The heaviest model runs first in each script. A failure that only shows up
# on the big backbones -- memory, most likely -- then surfaces in the first
# hour instead of a day later.
#
# Usage:
#   ./run_gpu1.sh                       # 240 epochs, eval and save every 10
#   ./run_gpu1.sh --epochs 50           # a shorter check
#   ./run_gpu1.sh --share /mnt/... # also copy results elsewhere
#   ./run_gpu1.sh --models dekr_w48     # just one
#
# Projects land in /data/disk2/home/tony/dlc_projects, one folder per model.
#
# Run both halves at once from the repository root:
#   ./scripts/dlc_train/run_gpu0.sh > gpu0.log 2>&1 &
#   ./scripts/dlc_train/run_gpu1.sh > gpu1.log 2>&1 &
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

GPU=1
CORES="12-23"
MODELS="dekr_w48,dekr_w18,resnet_101,hrnet_w32,dlcrnet_stride16_ms5"

cd "$REPO"
exec taskset -c "$CORES" \
    pixi run -e dlc python "$HERE/train_models.py" \
        --gpu "$GPU" --models "$MODELS" "$@"
