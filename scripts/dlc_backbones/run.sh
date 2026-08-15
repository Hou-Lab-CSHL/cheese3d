#!/usr/bin/env bash
# Time DeepLabCut's HRNet, DEKR, ResNet and DLCRNet backbones on one GPU.
#
#   hrnet_w18   9.56M    dekr_w18   9.56M
#   resnet_50  23.51M    dekr_w32  29.31M
#   hrnet_w32  29.31M    dekr_w48  65.33M
#   resnet_101 42.50M    hrnet_w48 65.33M
#   dlcrnet_stride32_ms5   dlcrnet_stride16_ms5   (ResNet-50 trunk)
#
# Answers two questions per model: does it run, and what would 300 epochs
# cost. The report also records seconds per epoch and peak GPU memory.
#
# Three epochs by default, not one. DeepLabCut prints no per-epoch duration,
# so the harness times the gaps between epoch boundaries in its log instead --
# which needs at least two epochs, and three for a median that ignores a slow
# first one. At --epochs 1 the 300-epoch column reads '?'.
#
# Usage:
#   ./run.sh                          # 10 models, 3 epochs, batch 32, GPU 0
#   ./run.sh --batch-size 16          # if batch 32 runs out of memory
#   ./run.sh --gpu 1
#   ./run.sh --models hrnet_w48,dekr_w48
#   ./run.sh --project-epochs 500     # project to a different epoch count
#   ./run.sh --fit-batch              # also fit the largest batch that fits
#
# Report location, in order of precedence: --report-dir, then
# $CHEESE3D_REPORT_DIR (which covers every suite at once), then
# <repo>/reports/dlc_backbones/. Filename is DLC_BACKBONES_RESULTS.md.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
EPOCHS=3
BATCH=32
GPU=0
REPORT_DIR="${CHEESE3D_REPORT_DIR:-$REPO/reports/dlc_backbones}"
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch-size) BATCH="$2";      shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -/{/^set -/!p;}' "$0"; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

cd "$REPO"
ROOT="$(dirname "$REPO")/cheese3d_dlc_backbone_tests"
REPORT="$REPORT_DIR/DLC_BACKBONES_RESULTS.md"
mkdir -p "$ROOT" "$REPORT_DIR"

{
    echo "# DeepLabCut backbone timings"
    echo
    echo "HRNet, DEKR, ResNet and DLCRNet on one GPU, batch ${BATCH},"
    echo "${EPOCHS} epoch(s) each, projected to 300."
    echo
    echo "Seconds per epoch is measured from the gaps between epoch boundaries"
    echo "in the training log, so it excludes startup and final evaluation."
    echo "The 300-epoch estimate adds that startup back once."
    echo
    echo "Generated: $(date '+%Y-%m-%d %H:%M')"
    echo
    echo '```'
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
        2>/dev/null || echo "nvidia-smi unavailable"
    echo '```'
} > "$REPORT"

echo "=============================================================="
echo "DLC backbones  (pixi env: dlc)"
echo "=============================================================="
pixi run -e dlc python "$HERE/test_dlc.py" \
    --epochs "$EPOCHS" --batch-size "$BATCH" --gpu "$GPU" --root "$ROOT" \
    --report-dir "$REPORT_DIR" --report-name "DLC_BACKBONES_RESULTS.md" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
status=$?

echo
echo "=============================================================="
[ $status -eq 0 ] && echo "every backbone ran" || echo "some backbones failed"
echo "report: $REPORT"
exit $status
