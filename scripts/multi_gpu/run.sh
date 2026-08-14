#!/usr/bin/env bash
# Multi-GPU test for the architectures whose behaviour above one card is not
# settled: SLEAP's convnext_large, SLEAP's unet_large_rf as the control, and
# Lightning Pose's DINO ViTs.
#
# Runs on two GPUs by default. Two is enough to answer the question these
# models pose -- whether they work above one card at all -- since the failure
# mode is a DDP deadlock, which appears at two ranks just as it does at four.
# The report records time, seconds per epoch, peak memory per card, and a
# projected 300-epoch run time.
#
# Usage:
#   ./run.sh                                  # 2 GPUs, 1 epoch, batch 16
#   ./run.sh --gpu 0,1,2,3                    # more cards, if a node has them
#   ./run.sh --batch-size 32 --epochs 3       # 3 epochs times per-epoch cost
#   ./run.sh --only sleap                     # one backend
#   ./run.sh --models convnext_large --only sleap
#   ./run.sh --timeout 900                    # fail a hang sooner
#   ./run.sh --report-dir /path/out           # this run only
#
# Report location, in order of precedence: --report-dir, then
# $CHEESE3D_REPORT_DIR (which covers every suite at once), then
# <repo>/reports/multi_gpu.
#
# Two things to know before reading the results:
#
#   * SLEAP batches per GPU under DDP, so --batch-size is per card there and
#     the effective batch is that times the GPU count. Lightning Pose batches
#     across the run, so its --batch-size is the effective batch as written.
#   * The known multi-GPU failure here is a spin-wait, not a crash: one GPU
#     pinned at 100% holding a little over a gigabyte, making no progress.
#     From outside that is indistinguishable from slow training until the
#     timeout fires, so it surfaces as TIMEOUT rather than ERROR. When a row
#     times out, check its log for whether an epoch boundary was ever reached.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
EPOCHS=1
GPU="0,1"
ONLY=""
REPORT_DIR="${CHEESE3D_REPORT_DIR:-$REPO/reports/multi_gpu}"
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        --only)       ONLY="$2";       shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -/{/^set -/!p;}' "$0"; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

cd "$REPO"
ROOT="$(dirname "$REPO")/cheese3d_multi_gpu_tests"
REPORT="$REPORT_DIR/MULTI_GPU_RESULTS.md"
mkdir -p "$ROOT" "$REPORT_DIR"

echo "GPUs visible to this node:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
    echo "nvidia-smi unavailable -- nothing to test"; exit 1; }

# Only a full run owns the whole report; --only refreshes just its backend.
if [ -z "$ONLY" ] || [ ! -f "$REPORT" ]; then
    {
        echo "# Multi-GPU test"
        echo
        echo "convnext_large, unet_large_rf and the DINO ViTs on GPUs ${GPU},"
        echo "${EPOCHS} epoch(s) each. Checks that these train above one card."
        echo
        echo "unet_large_rf is the control: at 1.70 M parameters it is known"
        echo "good on one GPU, so if it fails here too the problem is DDP or"
        echo "the environment rather than the model under test."
        echo
        echo "A TIMEOUT row is the one to read carefully. The known multi-GPU"
        echo "failure is an NCCL spin-wait, not a crash, and looks like slow"
        echo "training from outside. Check the log for an epoch boundary."
        echo
        echo "Lightning Pose skips its own evaluation for ViTs on more than one"
        echo "GPU, because that pass is where it deadlocks. An OK row for a ViT"
        echo "means training succeeded, not evaluation; predictions come from"
        echo "\`cheese3d track\`."
        echo
        echo "Generated: $(date '+%Y-%m-%d %H:%M')"
        echo
        echo '```'
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
        echo '```'
    } > "$REPORT"
fi

declare -a FAILED=()
run_backend () {   # run_backend <label> <pixi env> <script>
    local label="$1" env="$2" script="$3"
    [ -n "$ONLY" ] && [ "$ONLY" != "$label" ] && return 0
    echo
    echo "=============================================================="
    echo "$label  (pixi env: $env)"
    echo "=============================================================="
    pixi run -e "$env" python "$HERE/$script" \
        --epochs "$EPOCHS" --gpu "$GPU" --root "$ROOT" \
        --report-dir "$REPORT_DIR" --report-name "MULTI_GPU_RESULTS.md" \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        || FAILED+=("$label")
}

run_backend sleap sleap test_sleap.py
run_backend lp    lp    test_lightning_pose.py

echo
echo "=============================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "all backends: every model trained on ${GPU}"
else
    echo "backends with failures: ${FAILED[*]}"
fi
echo "report: $REPORT"
exit ${#FAILED[@]}
