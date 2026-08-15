#!/usr/bin/env bash
# Find the dataloader worker count that trains DeepLabCut fastest.
#
# One architecture, one GPU, one batch size; only the number of augmentation
# worker processes changes. DLC3 runs albumentations transforms per sample and
# defaults to zero workers, doing that work inside the training process while
# the GPU waits.
#
# Compare seconds per epoch, not total time -- startup does not scale with
# workers. GPU busy beside it says why: slow with a high percentage means
# compute-bound and more workers will not help; a low percentage means the
# card is still starved.
#
# Two epochs is enough. Per-epoch time is measured from the gap between epoch
# boundaries, so two epochs give one gap -- the duration of the second epoch,
# with warmup already behind it. A third would only average in more of the
# same at a third more GPU time.
#
# Usage:
#   ./run.sh                                    # -1,4,8,16,32,48 on hrnet_w18
#   ./run.sh --counts 0,24,48,64                # a different ladder
#   ./run.sh --architecture resnet_50           # a different model
#   ./run.sh --batch-size 16 --epochs 5
#
# workers=-1 forces DLC's own single-process default and is the baseline.
#
# Report location, in order of precedence: --report-dir, then
# $CHEESE3D_REPORT_DIR, then <repo>/reports/dlc_workers/. Filename is
# DLC_WORKERS_RESULTS.md.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
EPOCHS=2
BATCH=32
GPU=0
REPORT_DIR="${CHEESE3D_REPORT_DIR:-$REPO/reports/dlc_workers}"
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
ROOT="$(dirname "$REPO")/cheese3d_dlc_worker_tests"
REPORT="$REPORT_DIR/DLC_WORKERS_RESULTS.md"
mkdir -p "$ROOT" "$REPORT_DIR"

{
    echo "# DeepLabCut dataloader worker sweep"
    echo
    echo "One architecture, one GPU, batch ${BATCH}, ${EPOCHS} epoch(s)."
    echo "Only the augmentation worker count changes between rows."
    echo
    echo "Read seconds per epoch, not total time: startup does not scale with"
    echo "workers. \`workers=-1\` is DLC's own single-process default."
    echo
    echo "Cores available: $(nproc)"
    echo "Generated: $(date '+%Y-%m-%d %H:%M')"
    echo
    echo '```'
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
        2>/dev/null || echo "nvidia-smi unavailable"
    echo '```'
} > "$REPORT"

echo "=============================================================="
echo "DLC dataloader workers  (pixi env: dlc)"
echo "=============================================================="
pixi run -e dlc python "$HERE/test_workers.py" \
    --epochs "$EPOCHS" --batch-size "$BATCH" --gpu "$GPU" --root "$ROOT" \
    --report-dir "$REPORT_DIR" --report-name "DLC_WORKERS_RESULTS.md" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
status=$?

echo
echo "=============================================================="
[ $status -eq 0 ] && echo "every worker count ran" || echo "some runs failed"
echo "report: $REPORT"
exit $status
