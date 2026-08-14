#!/usr/bin/env bash
# Medium models, tier 1: backbones of 22-50 M parameters, on a single GPU.
#
#   dlc    resnet_50 23.51M  hrnet_w32 29.31M  dekr_w32 29.31M
#          resnet_101 42.50M  cspnext_x 47.57M
#          dlcrnet_stride32_ms5  dlcrnet_stride16_ms5
#   lp     vits_dino ~22M  resnet50 25.6M and its six pretrained variants
#          (animal_ap10k, animal_apose, human_hand, human_jhmdb,
#           human_res_rle, human_top_res)  resnet101 44.5M
#
# SLEAP has nothing in this range -- its backbones jump from 7.85 M straight
# to 87.66 M -- so it is not run here. Tier 2 covers all of them.
#
# 16 models. Expect the DLCRNet pair to fail: they train, then die in
# evaluation with "ValueError: [n, m] is not in list" from prune_paf_graph.
# They are kept in so that stays visible rather than being quietly dropped.
#
# Usage:
#   ./run_tier1.sh                     # 1 epoch, batch 16, GPU 0
#   ./run_tier1.sh --epochs 3          # times epochs properly on every backend
#   ./run_tier1.sh --fit-batch         # fit the max batch instead of bounding it
#   ./run_tier1.sh --only lp
#   ./run_tier1.sh --models resnet50,resnet101 --only lp
#
# Report location, in order of precedence: --report-dir, then
# $CHEESE3D_REPORT_DIR (which covers every suite at once), then
# <repo>/reports/medium_models/. Filename is MEDIUM_1_RESULTS.md.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIER=1
TIER_BACKENDS="dlc:dlc:test_dlc.py lp:lp:test_lightning_pose.py"
TIER_SUMMARY="Backbones of 22-50 M parameters, on one GPU. SLEAP is absent: it
has no backbone in this range."

source "$HERE/_tier.sh"
