#!/usr/bin/env bash
# End-to-end test of all three pose backends (DLC, Lightning Pose, SLEAP).
#
# For each backend this script:
#   1. creates a fresh Cheese3D project seeded from the DLC labels in
#      /data/disk2/home/tony/cheese3d2_test_set (weights + labeled frames),
#   2. copies in the test videos (hardlinked when possible, so no extra disk),
#   3. synchronizes the multi-camera videos,
#   4. trains the network: 100 epochs, batch 64, validate + save every 20
#      epochs, keep 5 checkpoints, motion blur OFF,
#   5. runs calibration, 2D tracking, 3D triangulation,
#   6. generates the labeled overlay videos.
#
# Backend notes on the requested settings:
#   - DLC:   motion blur is an explicit toggle -> off; max_snapshots_to_keep=5.
#   - LP:    motion blur is disabled by zeroing MotionBlur probability inside
#            the DLC-style augmentation preset (all other augmentations keep
#            their defaults). Lightning Pose has no keep-N-checkpoints option;
#            it retains the checkpoints written every 20 epochs.
#   - SLEAP: SLEAP-NN's augmentation has no motion blur at all (geometric
#            only), so nothing to disable; save_top_k=5 keeps the best five
#            checkpoints. Early stopping is disabled so all 100 epochs run.
#
# Network choice per backend -- the recommended architecture for fine-scale
# mouse facial keypoints (single animal, head-fixed, small features like eye
# corners and whisker-pad points):
#   - DLC:   hrnet_w32 -- HRNet keeps a high-resolution feature stream
#            end-to-end and outperforms resnet_50 on fine keypoints in DLC3's
#            PyTorch benchmarks; single-animal bottom-up variant (the
#            top_down_* variants need a separate detector stage).
#   - LP:    resnet50_animal_ap10k -- pretrained on the AP-10K animal-pose
#            dataset; Lightning Pose's recommended backbone for animal work.
#   - SLEAP: unet_medium_rf -- SLEAP's recommended single-instance baseline;
#            medium receptive field matches face-scale features at 640x512.
#
# Usage:
#   ./test_all_backends.sh [output_root]
#   GPU=0 ./test_all_backends.sh              # restrict to a single GPU
#   ONLY=sleap ./test_all_backends.sh         # test a single backend
#
# Training runs on both GPUs by default (GPU="0,1"); SLEAP trains with DDP,
# so its batch size is per GPU (64 x 2 = 128 effective).
#
set -uo pipefail

REPO="/data/disk2/home/tony/cheese3d"
TESTSET="/data/disk2/home/tony/cheese3d2_test_set"
DLC_SOURCE="$TESTSET/cheese3d_demo_model-houlab-2025-05-28"
VIDEO_SOURCE="$TESTSET/test_vid/20231031_chew"
ROOT="${1:-/data/disk2/home/tony/cheese3d_backend_tests}"
GPU="${GPU:-0,1}"
ONLY="${ONLY:-}"

cd "$REPO"

[ -d "$DLC_SOURCE" ] || { echo "missing DLC source: $DLC_SOURCE" >&2; exit 1; }
[ -d "$VIDEO_SOURCE" ] || { echo "missing test videos: $VIDEO_SOURCE" >&2; exit 1; }
mkdir -p "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Training settings JSON per backend. Common: 100 epochs, batch 64,
# validation + checkpoint every 20 epochs. Learning rates are each backend's
# Cheese3D GUI default.
# ---------------------------------------------------------------------------

DLC_SETTINGS='{
  "epochs": 100, "batch_size": 64, "learning_rate": 0.0005,
  "save_every_n_epochs": 20, "validate_every_n_epochs": 20,
  "network_architecture": "hrnet_w32",
  "train_fraction_percent": 95, "training_shuffle": 1,
  "max_snapshots_to_keep": 5,
  "rotation": 30, "scale_min": 0.5, "scale_max": 1.25,
  "crop_width": 448, "crop_height": 448,
  "motion_blur": false,
  "gaussian_noise": 12.75
}'

# DLC-style augmentation with MotionBlur probability forced to 0; every other
# transform keeps the Cheese3D GUI default.
LP_SETTINGS='{
  "epochs": 100, "batch_size": 64, "learning_rate": 0.001,
  "save_every_n_epochs": 20, "validate_every_n_epochs": 20,
  "backbone": "resnet50_animal_ap10k",
  "horizontal_flip": false,
  "train_prob": 0.95, "val_prob": 0.05,
  "unfreezing_epoch": 20,
  "early_stopping": false, "early_stop_patience": 3,
  "imgaug": {
    "Affine": {"p": 0.4, "kwargs": {"rotate": [-25, 25]}},
    "MotionBlur": {"p": 0.0, "kwargs": {"k": 5, "angle": [-90, 90]}},
    "CoarseDropout": {"p": 0.5, "kwargs": {"p": 0.02, "size_percent": 0.3, "per_channel": 0.5}},
    "CoarseSalt": {"p": 0.5, "kwargs": {"p": 0.01, "size_percent": [0.05, 0.1]}},
    "CoarsePepper": {"p": 0.5, "kwargs": {"p": 0.01, "size_percent": [0.05, 0.1]}},
    "ElasticTransformation": {"p": 0.5, "kwargs": {"alpha": [0, 10], "sigma": 5}},
    "AllChannelsHistogramEqualization": {"p": 0.1, "kwargs": {}},
    "AllChannelsCLAHE": {"p": 0.1, "kwargs": {}},
    "Emboss": {"p": 0.1, "kwargs": {"alpha": [0.0, 0.5], "strength": [0.5, 1.5]}},
    "CropAndPad": {"p": 0.4, "kwargs": {"percent": [-0.15, 0.15], "keep_size": false}}
  }
}'

SLEAP_SETTINGS='{
  "epochs": 100, "batch_size": 64, "learning_rate": 0.0001,
  "save_every_n_epochs": 20, "validate_every_n_epochs": 20,
  "backbone": "unet_medium_rf",
  "validation_fraction_percent": 10, "val_batch_size": 4,
  "optimizer": "Adam",
  "min_steps_per_epoch": 200, "steps_per_epoch": 0,
  "save_top_k": 5, "save_last": true,
  "early_stopping": false, "early_stop_patience": 10,
  "use_augmentation": true,
  "rotation_min": -15, "rotation_max": 15,
  "scale_min": 0.9, "scale_max": 1.1,
  "translate": 0.0
}'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

patch_config() {  # patch_config <project_dir> <backend_type>
    local project_dir="$1" backend_type="$2"
    local session_name
    session_name="$(basename "$VIDEO_SOURCE")"
    pixi run -e dlc python - "$project_dir" "$backend_type" "$DLC_SOURCE" "$session_name" <<'PYEOF'
import sys
import yaml

project_dir, backend_type, dlc_source, session_name = sys.argv[1:5]
config_path = f"{project_dir}/config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cfg["model"]["name"] = "cheese3d_demo_model"
cfg["model"]["backend_type"] = backend_type
cfg["model"]["backend_options"] = {
    "source_project_path": dlc_source,
    "source_format": "dlc",
}
# find_videos only scans sessions declared here -- the default (empty) list
# would silently discover no videos at all.
cfg["sessions"] = [{"name": session_name}]

with open(config_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"patched {config_path}: backend_type={backend_type}, session={session_name}")
PYEOF
}

copy_videos() {  # copy_videos <project_dir>
    local dest="$1/videos"
    mkdir -p "$dest"
    # Hardlink when source and destination share a filesystem (instant, no
    # extra space); fall back to a real copy otherwise.
    cp -al "$VIDEO_SOURCE" "$dest/" 2>/dev/null || cp -a "$VIDEO_SOURCE" "$dest/"
}

run_step() {  # run_step <log_file> <description> <command...>
    local log="$1" desc="$2"; shift 2
    echo "  -> $desc"
    local start end
    start=$(date +%s)
    if "$@" >> "$log" 2>&1; then
        end=$(date +%s)
        echo "     done in $((end - start))s"
        return 0
    else
        end=$(date +%s)
        echo "     FAILED after $((end - start))s (see $log)" >&2
        return 1
    fi
}

run_backend() {  # run_backend <backend_type> <pixi_env> <settings_json>
    local backend_type="$1" env="$2" settings="$3"
    local project="c3d_${backend_type}"
    local project_dir="$ROOT/$project"
    local log="$LOG_DIR/${project}.log"
    local c3d=(pixi run -e "$env" cheese3d --path "$ROOT")

    echo "=============================================================="
    echo "Backend: $backend_type (pixi env: $env)"
    echo "Project: $project_dir"
    echo "Log:     $log"
    echo "=============================================================="
    : > "$log"

    if [ -e "$project_dir" ]; then
        echo "  project dir already exists -- remove it to rerun: $project_dir" >&2
        return 1
    fi

    run_step "$log" "create project" \
        "${c3d[@]}" setup "$project" || return 1
    run_step "$log" "configure backend + import DLC labels" \
        patch_config "$project_dir" "$backend_type" || return 1
    run_step "$log" "copy test videos" \
        copy_videos "$project_dir" || return 1
    run_step "$log" "summarize (validates project, builds backend)" \
        "${c3d[@]}" summarize "$project" || return 1
    run_step "$log" "synchronize videos" \
        "${c3d[@]}" sync "$project" || return 1
    run_step "$log" "train (100 epochs, batch 64, val/save every 20, no motion blur)" \
        "${c3d[@]}" train "$project" --gpu "$GPU" --training-settings "$settings" || return 1
    run_step "$log" "calibrate" \
        "${c3d[@]}" calibrate "$project" || return 1
    run_step "$log" "track 2D keypoints" \
        "${c3d[@]}" track "$project" || return 1
    run_step "$log" "triangulate 3D" \
        "${c3d[@]}" triangulate "$project" || return 1
    run_step "$log" "generate labeled videos" \
        "${c3d[@]}" generate-videos "$project" || return 1

    echo "  $backend_type: ALL STEPS PASSED"
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

declare -A RESULTS
overall_start=$(date +%s)

for spec in "dlc:dlc:$DLC_SETTINGS" \
            "lightning_pose:lp:$LP_SETTINGS" \
            "sleap:sleap:$SLEAP_SETTINGS"; do
    backend_type="${spec%%:*}"
    rest="${spec#*:}"
    env="${rest%%:*}"
    settings="${rest#*:}"

    if [ -n "$ONLY" ] && [ "$ONLY" != "$backend_type" ]; then
        RESULTS[$backend_type]="SKIPPED"
        continue
    fi

    if run_backend "$backend_type" "$env" "$settings"; then
        RESULTS[$backend_type]="PASSED"
    else
        RESULTS[$backend_type]="FAILED"
    fi
done

overall_end=$(date +%s)
echo
echo "=============================================================="
echo "Summary ($(( (overall_end - overall_start) / 60 )) min total)"
echo "=============================================================="
exit_code=0
for backend_type in dlc lightning_pose sleap; do
    printf "  %-16s %s\n" "$backend_type" "${RESULTS[$backend_type]:-NOT RUN}"
    [ "${RESULTS[$backend_type]:-}" = "FAILED" ] && exit_code=1
done
echo "Projects under: $ROOT"
echo "Logs under:     $LOG_DIR"
exit "$exit_code"
