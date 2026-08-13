#!/usr/bin/env bash
# End-to-end test of all three pose backends (DLC, Lightning Pose, SLEAP).
#
# For each backend this script:
#   1. creates a fresh Cheese3D project seeded from the DLC labels in
#      /data/disk2/home/tony/cheese3d2_test_set (weights + labeled frames),
#   2. copies in the test videos (hardlinked when possible, so no extra disk),
#   3. trains the network: 5 epochs, validate + save every epoch, keep 5
#      checkpoints, motion blur OFF. Five epochs is a pipeline smoke test,
#      not a quality run -- per-epoch checkpointing exists so the tracking
#      step has something to load.
#   4. runs calibration and 2D tracking,
#   5. generates the labeled 2D overlay videos.
#
# 3D triangulation is skipped: generate_videos_2d reads pose-2d only, and the
# 3D-reprojection branch of generate_videos is already guarded by a check for
# pose-3d CSVs, so it is simply not taken.
#
# Video synchronization is deliberately skipped: it only writes .align.json
# and QC PNGs, and nothing downstream (tracking, triangulation, video
# generation) reads them -- confirmed by searching the whole codebase for
# consumers. Run `cheese3d sync <project>` separately if you want them.
#
# Backend notes on the requested settings:
#   - DLC:   motion blur is an explicit toggle -> off; max_snapshots_to_keep=5.
#   - LP:    motion blur is disabled by zeroing MotionBlur probability inside
#            the DLC-style augmentation preset (all other augmentations keep
#            their defaults). Lightning Pose has no keep-N-checkpoints option;
#            it retains the checkpoints written every 20 epochs.
#   - SLEAP: SLEAP-NN's augmentation has no motion blur at all (geometric
#            only), so nothing to disable; save_top_k=5 keeps the best five
#            checkpoints. Early stopping is disabled so every epoch runs.
#            min_steps_per_epoch used to force a floor of 200 optimizer steps
#            per epoch regardless of dataset size, inflating each epoch ~15x
#            over a natural pass. Cheese3D now defaults that floor to 0, so a
#            SLEAP epoch means one dataset pass, as it does for DLC and LP.
#
# Network choice per backend -- the recommended architecture for fine-scale
# mouse facial keypoints (single animal, head-fixed, small features like eye
# corners and whisker-pad points):
#   - DLC:   resnet_50 -- DLC3's default and best-validated PyTorch backbone,
#            and the one its default learning rate is tuned for. hrnet_w32
#            was tried first and its loss stayed flat (validation RMSE 419 px,
#            mAP 0.00, i.e. random) for 30+ epochs at this learning rate.
#   - LP:    resnet50_animal_ap10k -- pretrained on the AP-10K animal-pose
#            dataset; Lightning Pose's recommended backbone for animal work.
#   - SLEAP: convnext_tiny -- the smallest backbone SLEAP-NN can load ImageNet
#            weights for. unet_medium_rf (SLEAP's nominal baseline) was tried
#            first and collapsed: SLEAP-NN has no pretrained UNet, and from
#            random init on 860 images the model learned to predict background
#            everywhere -- 97.7% of inferred points landed on the image border
#            at ~0.035 confidence, under SLEAP's 0.2 detection threshold, so
#            every 2D output was NaN and 3D triangulation died on all-NaN
#            input. Pretraining is what makes DLC's resnet_50 work here too.
#
# Usage:
#   ./test_all_backends.sh [output_root]
#   GPU=0 ./test_all_backends.sh              # restrict to a single GPU
#   ONLY=sleap ./test_all_backends.sh         # test a single backend
#
# Training runs on both GPUs by default (GPU="0,1"). SLEAP trains with DDP,
# where batch size is per GPU (40 x 2 GPUs = 80 effective). DLC and LP take
# the batch size as given.
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
# Training settings JSON per backend. Common: 5 epochs, validation and
# checkpointing every epoch. Learning rates are each backend's Cheese3D GUI
# default. Batch differs by backend -- see the SLEAP note below for why.
# ---------------------------------------------------------------------------

DLC_SETTINGS='{
  "epochs": 5, "batch_size": 64, "learning_rate": 0.0005,
  "save_every_n_epochs": 1, "validate_every_n_epochs": 1,
  "network_architecture": "resnet_50",
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
  "epochs": 5, "batch_size": 64, "learning_rate": 0.001,
  "save_every_n_epochs": 1, "validate_every_n_epochs": 1,
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

# The memory fix here is input_scale, not batch size. SLEAP runs its encoder
# over whole 640x512 camera views (it is normally pointed at small crops), and
# at scale 1.0 that OOMed these 47.4 GiB cards at every batch tried -- 64, 32
# and 16 per GPU, at head/backbone output_stride 1 and 2 alike. Halving the
# input quarters the activation area: measured 9.3 GiB at batch 8/GPU, where
# the same run at scale 1.0 could not start. SLEAP-NN rescales labels with the
# images and restores original-image coordinates at inference.
#
# batch_size is PER GPU under SLEAP's DDP (trainer_devices = number of GPUs),
# so 40 here is an effective batch of 80 -- larger than DLC's and LP's 64.
# Measured peaks per GPU at scale 0.5: 9.3 GiB at batch 8, ~41 GiB at batch 48
# (88% of the card, too tight to leave unattended). 40 sits near 36 GiB and
# leaves room for the periodic validation pass, which val_batch_size keeps
# deliberately small so it cannot spike into the ceiling.
SLEAP_SETTINGS='{
  "epochs": 5, "batch_size": 40, "learning_rate": 0.0001,
  "input_scale": 0.5,
  "save_every_n_epochs": 1, "validate_every_n_epochs": 1,
  "backbone": "convnext_tiny",
  "validation_fraction_percent": 10, "val_batch_size": 8,
  "optimizer": "Adam",
  "min_steps_per_epoch": 0, "steps_per_epoch": 0,
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

verify_predictions() {  # verify_predictions <project_dir> <env>
    # Every pipeline step can exit 0 while the pipeline as a whole produces
    # nothing: a model whose peak confidences all fall below the backend's
    # detection threshold yields an entirely NaN pose table, tracking still
    # reports success, and video generation happily renders clips with no
    # keypoints drawn. That happened here -- SLEAP reported ALL STEPS PASSED
    # with NaN fraction 1.000 -- so step completion alone is not evidence the
    # backend works. Fail loudly rather than reporting a green run.
    local project_dir="$1" env="$2"
    pixi run -e "$env" python - "$project_dir" <<'PYEOF'
import glob
import sys

import numpy as np
import pandas as pd

files = sorted(glob.glob(f"{sys.argv[1]}/triangulation/*/pose-2d/*.h5"))
if not files:
    print("no 2D pose tables were written")
    raise SystemExit(1)

worst = 0.0
for path in files:
    frame = pd.read_hdf(path)
    nan_fraction = float(np.isnan(frame.values.astype(float)).mean())
    worst = max(worst, nan_fraction)
    likelihood = frame.xs("likelihood", level=-1, axis=1).values.astype(float)
    median = "all-NaN" if np.isnan(likelihood).all() else f"{np.nanmedian(likelihood):.3f}"
    print(f"  {path.rsplit('/', 1)[-1][:52]:<52} NaN {nan_fraction:.3f}  median p {median}")

if worst == 1.0:
    print("every keypoint is NaN: the model predicted nothing above its "
          "detection threshold, so the labeled videos contain no keypoints")
    raise SystemExit(1)
print(f"predictions contain data (worst NaN fraction {worst:.3f})")
PYEOF
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

    # A backend that already completed every step is never redone, so this
    # script can be rerun after fixing a failure without repeating the
    # multi-hour training runs that already succeeded.
    if [ -f "$project_dir/.cheese3d_test_passed" ]; then
        echo "  already completed -- skipping (delete $project_dir to redo)"
        return 0
    fi

    # A partially-completed project is rebuilt from scratch: the failed step
    # may have left the backend/config half-written, and every step before
    # training is cheap.
    if [ -e "$project_dir" ]; then
        echo "  removing incomplete project from a previous run"
        rm -rf "$project_dir"
    fi

    : > "$log"

    run_step "$log" "create project" \
        "${c3d[@]}" setup "$project" || return 1
    run_step "$log" "configure backend + import DLC labels" \
        patch_config "$project_dir" "$backend_type" || return 1
    run_step "$log" "copy test videos" \
        copy_videos "$project_dir" || return 1
    run_step "$log" "summarize (validates project, builds backend)" \
        "${c3d[@]}" summarize "$project" || return 1
    run_step "$log" "train (5 epochs, val/save every epoch, no motion blur)" \
        "${c3d[@]}" train "$project" --gpu "$GPU" --training-settings "$settings" || return 1
    run_step "$log" "calibrate" \
        "${c3d[@]}" calibrate "$project" || return 1
    run_step "$log" "track 2D keypoints" \
        "${c3d[@]}" track "$project" || return 1
    run_step "$log" "generate labeled videos" \
        "${c3d[@]}" generate-videos "$project" || return 1
    run_step "$log" "verify 2D predictions contain data" \
        verify_predictions "$project_dir" "$env" || return 1

    touch "$project_dir/.cheese3d_test_passed"
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
