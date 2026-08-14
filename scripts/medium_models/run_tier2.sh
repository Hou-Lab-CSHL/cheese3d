#!/usr/bin/env bash
# Medium models, tier 2: backbones above 50 M parameters, on a single GPU.
#
#   dlc    hrnet_w48 65.33M  dekr_w48 65.33M
#   lp     resnet152 60.2M  vitb_dino ~86M  vitb_imagenet ~86M
#   sleap  swint_tiny 87.66M     convnext_tiny 87.96M
#          swint_small 108.97M   convnext_small 109.59M
#          swint_base 193.65M    convnext_base 194.47M
#
# 11 models. The SLEAP names are misleading: "tiny" and "small" describe the
# ImageNet classifiers these wrap, and once SLEAP adds its decoder none of
# them are either. Counts above are measured, not quoted.
#
# This is the tier where memory decides the outcome. SLEAP encodes whole
# frames, so --input-scale drives peak memory harder than the backbone does;
# convnext_base may not fit at batch 16 even on an 80 GB card. Lower
# --batch-size until it does, or take the OOM as the answer -- it is a result,
# not a bug to chase.
#
# Usage:
#   ./run_tier2.sh                     # 1 epoch, batch 16, GPU 0
#   ./run_tier2.sh --batch-size 8      # if the big SLEAP backbones OOM
#   ./run_tier2.sh --epochs 3          # times epochs properly on every backend
#   ./run_tier2.sh --fit-batch         # fit the max batch instead of bounding it
#   ./run_tier2.sh --only sleap
#   ./run_tier2.sh --models convnext_base --only sleap
#
# Report location, in order of precedence: --report-dir, then
# $CHEESE3D_REPORT_DIR (which covers every suite at once), then
# <repo>/reports/medium_models/. Filename is MEDIUM_2_RESULTS.md.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIER=2
TIER_BACKENDS="dlc:dlc:test_dlc.py lp:lp:test_lightning_pose.py sleap:sleap:test_sleap.py"
TIER_SUMMARY="Backbones above 50 M parameters, on one GPU. This is where memory,
not parameter count, decides whether a model runs."

source "$HERE/_tier.sh"
