#!/usr/bin/env bash
# DLC training, half of the models, GPU 0, cores 0-11.
#
#   hrnet_w48   65.33M   batch 16     the most expensive model in the set
#   hrnet_w18    9.56M   batch 32
#   resnet_50   23.51M   batch 32
#   dekr_w32    29.31M   batch 24
#   dlcrnet_stride32_ms5  batch 24
#
# Paired with run_gpu1.sh, which takes the other five on GPU 1 and cores
# 12-23. The two halves are balanced by measured cost, not by model count:
# hrnet_w18 alone projects to about 11 h at 300 epochs, and the W48 pair is
# far heavier, so one W48 goes to each script rather than both to one.
#
# The heaviest model runs first in each script. A failure that only shows up
# on the big backbones -- memory, most likely -- then surfaces in the first
# hour instead of a day later.
#
# Usage:
#   ./run_gpu0.sh                       # 240 epochs, eval and save every 10
#   ./run_gpu0.sh --epochs 50           # a shorter check
#   ./run_gpu0.sh --share /mnt/... # also copy results elsewhere
#   ./run_gpu0.sh --models hrnet_w48    # just one
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

GPU=0
CORES="0-11"
MODELS="hrnet_w48,hrnet_w18,resnet_50,dekr_w32,dlcrnet_stride32_ms5"

cd "$REPO"
exec taskset -c "$CORES" \
    pixi run -e dlc python "$HERE/train_models.py" \
        --gpu "$GPU" --models "$MODELS" "$@"
