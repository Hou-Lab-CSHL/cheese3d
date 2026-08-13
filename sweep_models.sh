#!/usr/bin/env bash
# Train every architecture each backend offers for one epoch, and record which
# ones work.
#
# This is a compatibility sweep, not a quality benchmark: one epoch proves the
# architecture can be constructed, fed this project's data, and take optimizer
# steps without crashing or exhausting the GPU. It says nothing about accuracy.
#
# Each backend gets its own throwaway project so the sweep never overwrites
# checkpoints from test_all_backends.sh. Batch size and input scale are held
# constant within a backend so a failure is attributable to the architecture
# rather than to differing settings; the values used are recorded in the report.
#
# Usage:
#   ./sweep_models.sh                     # all three backends
#   ONLY=sleap ./sweep_models.sh          # one backend
#   TIMEOUT=900 ./sweep_models.sh         # per-model timeout, default 900s
#
set -uo pipefail

# Paths are derived from this script's own location so the harness runs on any
# machine; override TESTSET (labelled source project + test videos) and the
# output root for a different data layout.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTSET="${TESTSET:-$(dirname "$REPO")/cheese3d2_test_set}"
DLC_SOURCE="$TESTSET/cheese3d_demo_model-houlab-2025-05-28"
VIDEO_SOURCE="$TESTSET/test_vid/20231031_chew"
ROOT="${1:-$(dirname "$REPO")/cheese3d_model_sweep}"
GPU="${GPU:-0,1}"
ONLY="${ONLY:-}"
TIMEOUT="${TIMEOUT:-900}"

cd "$REPO"
mkdir -p "$ROOT/logs"
REPORT="$ROOT/MODEL_SWEEP_RESULTS.md"

# --- architecture lists, read from the code so they cannot drift -------------
dlc_models=$(pixi run -e dlc python -c "
from cheese3d.backends.dlc import DLC3_PYTORCH_MODELS as M; print(' '.join(M))" 2>/dev/null | tail -1)
lp_models=$(pixi run -e lp python -c "
from cheese3d.interactive import LIGHTNING_POSE_BACKBONES as B; print(' '.join(B))" 2>/dev/null | tail -1)
sleap_models=$(pixi run -e sleap python -c "
from cheese3d.backends.sleap import SLEAP_BACKBONES as B; print(' '.join(B))" 2>/dev/null | tail -1)

build_project () {  # build_project <backend_type> <env> <project>
    local backend_type="$1" env="$2" project="$3"
    local dir="$ROOT/$project"
    [ -f "$dir/config.yaml" ] && return 0
    rm -rf "$dir"
    pixi run -e "$env" cheese3d --path "$ROOT" setup "$project" >/dev/null 2>&1 || return 1
    pixi run -e "$env" python - "$dir" "$backend_type" "$DLC_SOURCE" <<'PYEOF' >/dev/null 2>&1 || return 1
import sys, yaml
d, bt, src = sys.argv[1:4]
p = f"{d}/config.yaml"; cfg = yaml.safe_load(open(p))
cfg["model"]["name"] = "sweep_model"
cfg["model"]["backend_type"] = bt
cfg["model"]["backend_options"] = {"source_project_path": src, "source_format": "dlc"}
cfg["sessions"] = [{"name": "20231031_chew"}]
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
PYEOF
    mkdir -p "$dir/videos"
    cp -al "$VIDEO_SOURCE" "$dir/videos/" 2>/dev/null || cp -a "$VIDEO_SOURCE" "$dir/videos/"
    pixi run -e "$env" cheese3d --path "$ROOT" summarize "$project" >/dev/null 2>&1 || return 1
    return 0
}

classify () {  # classify <logfile> <exit_code>
    local log="$1" rc="$2"
    if grep -aq "OutOfMemoryError" "$log" 2>/dev/null; then echo "OOM"
    elif [ "$rc" = "124" ]; then echo "TIMEOUT"
    elif grep -aqiE "is unavailable|Unsupported .* backbone|not available" "$log" 2>/dev/null; then echo "UNAVAILABLE"
    elif [ "$rc" != "0" ]; then echo "ERROR"
    else echo "OK"
    fi
}

reason () {  # reason <logfile> <status>
    local log="$1" status="$2"
    case "$status" in
      OK) echo "-" ;;
      OOM) grep -aoE "Tried to allocate [0-9.]+ [GM]iB" "$log" 2>/dev/null | tail -1 ;;
      TIMEOUT) echo "exceeded ${TIMEOUT}s" ;;
      *) grep -aoE "[A-Za-z_.]*(Error|Exception)[^\"]{0,90}" "$log" 2>/dev/null \
           | grep -av "OutOfMemory" | tail -1 | tr -d '|' | cut -c1-110 ;;
    esac
}

sweep_backend () {  # sweep_backend <backend_type> <env> <models> <settings_fn>
    local backend_type="$1" env="$2" models="$3" settings_fn="$4"
    local project="sweep_${backend_type}"

    echo "=============================================================="
    echo "Sweeping $backend_type ($(echo "$models" | wc -w) architectures)"
    echo "=============================================================="
    if ! build_project "$backend_type" "$env" "$project"; then
        echo "  could not build $backend_type project -- skipping backend" >&2
        return 1
    fi

    echo "" >> "$REPORT"
    echo "## $backend_type" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Architecture | Result | Time | Detail |" >> "$REPORT"
    echo "|---|---|---:|---|" >> "$REPORT"

    for model in $models; do
        local log="$ROOT/logs/${backend_type}_${model}.log"
        local settings; settings="$($settings_fn "$model")"
        local start end rc status det
        printf "  %-28s " "$model"
        start=$(date +%s)
        timeout "$TIMEOUT" pixi run -e "$env" cheese3d --path "$ROOT" \
            train "$project" --gpu "$GPU" --no-iterate-dataset \
            --training-settings "$settings" > "$log" 2>&1
        rc=$?
        end=$(date +%s)
        status=$(classify "$log" "$rc")
        det=$(reason "$log" "$status")
        printf "%-12s %4ss\n" "$status" "$((end - start))"
        echo "| \`$model\` | $status | $((end - start))s | ${det:-–} |" >> "$REPORT"
        pkill -9 -f "envs/$env/bin/python" 2>/dev/null
        sleep 4
    done
}

dlc_settings () {
    cat <<EOF
{"epochs": 1, "batch_size": 64, "learning_rate": 0.0005,
 "save_every_n_epochs": 1, "validate_every_n_epochs": 1,
 "network_architecture": "$1", "train_fraction_percent": 95,
 "training_shuffle": 1, "max_snapshots_to_keep": 1,
 "rotation": 30, "scale_min": 0.5, "scale_max": 1.25,
 "crop_width": 448, "crop_height": 448,
 "motion_blur": false, "gaussian_noise": 12.75}
EOF
}

lp_settings () {
    cat <<EOF
{"epochs": 1, "batch_size": 64, "learning_rate": 0.001,
 "save_every_n_epochs": 1, "validate_every_n_epochs": 1,
 "backbone": "$1", "imgaug": "default", "horizontal_flip": false,
 "train_prob": 0.95, "val_prob": 0.05, "unfreezing_epoch": 20,
 "early_stopping": false, "early_stop_patience": 3}
EOF
}

sleap_settings () {
    cat <<EOF
{"epochs": 1, "batch_size": 16, "learning_rate": 0.0001, "input_scale": 0.5,
 "save_every_n_epochs": 1, "validate_every_n_epochs": 1,
 "backbone": "$1", "validation_fraction_percent": 10, "val_batch_size": 8,
 "optimizer": "Adam", "min_steps_per_epoch": 0, "steps_per_epoch": 0,
 "save_top_k": 1, "save_last": true,
 "early_stopping": false, "early_stop_patience": 10,
 "use_augmentation": true, "rotation_min": -15, "rotation_max": 15,
 "scale_min": 0.9, "scale_max": 1.1, "translate": 0.0}
EOF
}

# --- report header ----------------------------------------------------------
cat > "$REPORT" <<EOF
# Model compatibility sweep

One epoch per architecture. This proves each model can be built, fed this
project's data, and take optimizer steps without crashing or exhausting the
GPU. It is **not** an accuracy comparison.

Generated: $(date '+%Y-%m-%d %H:%M')
Hardware: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x $(nvidia-smi --list-gpus | wc -l)
Dataset: $(find "$DLC_SOURCE/labeled-data" -name '*.png' | wc -l) labeled images, 640x512

Settings held constant within each backend, so a failure is attributable to the
architecture rather than the configuration:

| Backend | Batch | Input scale | Notes |
|---|---|---|---|
| dlc | 64 | 1.0 | 448x448 random crop |
| lightning_pose | 64 | 1.0 | default (resize-only) augmentation |
| sleap | 16/GPU | 0.5 | batch is per GPU under DDP |

Result codes: **OK** trained one epoch · **OOM** exhausted GPU memory ·
**ERROR** crashed · **TIMEOUT** exceeded ${TIMEOUT}s · **UNAVAILABLE** not
installed in this environment.
EOF

start_all=$(date +%s)
[ -z "$ONLY" ] || [ "$ONLY" = "dlc" ] && sweep_backend dlc dlc "$dlc_models" dlc_settings
[ -z "$ONLY" ] || [ "$ONLY" = "lightning_pose" ] && sweep_backend lightning_pose lp "$lp_models" lp_settings
[ -z "$ONLY" ] || [ "$ONLY" = "sleap" ] && sweep_backend sleap sleap "$sleap_models" sleap_settings
end_all=$(date +%s)

{
  echo ""
  echo "Sweep completed in $(( (end_all - start_all) / 60 )) minutes."
} >> "$REPORT"

echo
echo "Report: $REPORT"
grep -cE "^\| \`.*\| OK \|" "$REPORT" | xargs -I{} echo "models OK: {}"
grep -E "^\| \`.*\| (OOM|ERROR|TIMEOUT|UNAVAILABLE) \|" "$REPORT" | sed 's/^/  /'
