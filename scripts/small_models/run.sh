#!/usr/bin/env bash
# Smoke-test every architecture smaller than ResNet-50 (25.6 M parameters),
# across all three backends.
#
# Each backend runs under its own Pixi environment, because they cannot share
# a Python process -- the Lightning Pose environment has no DeepLabCut
# installed and vice versa. This script is only the orchestrator; the actual
# test lives in the per-backend Python modules beside it.
#
# One epoch by default: enough to prove a model builds, accepts this project's
# data, and takes optimizer steps. Raise --epochs when you also want a signal
# about whether it learns.
#
# The 12 qualifying models, with measured parameter counts:
#
#   dlc    cspnext_s 4.38M  hrnet_w18 9.56M  dekr_w18 9.56M  cspnext_m 12.28M
#   lp     efficientnet_b0/b1/b2 5.3/7.8/9.1M  resnet18 11.7M  resnet34 21.8M
#   sleap  unet 1.33M  unet_large_rf 1.70M  unet_medium_rf 7.85M
#
# Usage:
#   ./run.sh                          # all three backends, 1 epoch, batch 16
#   ./run.sh --batch-size 64          # bigger batch (see note below)
#   ./run.sh --epochs 300             # the real training run
#   ./run.sh --only lp                # one backend
#   ./run.sh --gpu 0,1                # both GPUs
#   ./run.sh --models resnet18,resnet34 --only lp
#   ./run.sh --target-gb 80           # extrapolate batches to an 80 GB card
#   ./run.sh --report-dir /path/out   # write the report elsewhere
#
# Where the report goes, in order of precedence:
#   --report-dir /path/out            this run only
#   export CHEESE3D_REPORT_DIR=/path  every suite, until unset
#   <repo>/reports/small_models       the default
# It is kept out of the project tree either way, because that tree is deleted
# and rebuilt on every run. The suites write distinct filenames, so pointing
# them all at one directory does not collide.
#
# The report records wall-clock time and peak GPU memory per model, plus a
# conservative batch estimate for a --target-gb card (default 90). Read the
# note at the end of the report before trusting that estimate.
#
# --batch-size means samples per GPU for SLEAP, which batches per-device under
# DDP, and samples per step for DLC and Lightning Pose. On two GPUs, SLEAP's
# effective batch is therefore twice what you pass and the other two backends'
# is exactly what you pass.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
EPOCHS=1
GPU=0
ONLY=""
REPORT_DIR="${CHEESE3D_REPORT_DIR:-$REPO/reports/small_models}"
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
ROOT="$(dirname "$REPO")/cheese3d_small_model_tests"
# The report lives apart from the projects: build_project wipes and rebuilds
# its tree on every run, and results should outlive that.
REPORT="$REPORT_DIR/SMALL_MODEL_RESULTS.md"
mkdir -p "$ROOT" "$REPORT_DIR"
# Only a full run owns the whole report. With --only, the other backends'
# sections stay as they were and just this backend's rows get refreshed.
if [ -z "$ONLY" ] || [ ! -f "$REPORT" ]; then
    {
        echo "# Small-model smoke test"
        echo
        echo "Architectures below ResNet-50's 25.6 M parameters, ${EPOCHS} epoch(s) each."
        echo "This checks that a model builds, accepts the data and takes optimizer"
        echo "steps -- not that it learns anything."
        echo
        echo "Generated: $(date '+%Y-%m-%d %H:%M')"
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
        --report-dir "$REPORT_DIR" --report-name "SMALL_MODEL_RESULTS.md" \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        || FAILED+=("$label")
}

run_backend dlc   dlc   test_dlc.py
run_backend lp    lp    test_lightning_pose.py
run_backend sleap sleap test_sleap.py

echo
echo "=============================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "all backends: every small model ran"
else
    echo "backends with failures: ${FAILED[*]}"
fi
echo "report: $REPORT"
exit ${#FAILED[@]}
