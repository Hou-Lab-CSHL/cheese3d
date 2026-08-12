# Cheese3D Backend Integration: Current Status and Handoff

Last updated: 2026-08-11

This document is the current handoff point for continuing the Cheese3D backend work. It describes what is implemented in this working tree, how the environments are separated, how projects are laid out, what has been validated, and what remains unfinished.

> Important: the working tree currently contains uncommitted backend/config synchronization changes. Preserve these changes and inspect `git diff` before modifying overlapping files.

## Goal

Cheese3D should support three interchangeable 2-D pose-estimation backends while retaining the same downstream multi-camera tracking, triangulation, quality-control, and visualization workflow:

- DeepLabCut 3 (DLC3)
- Lightning Pose
- SLEAP

Each backend has incompatible or sensitive Python/CUDA dependencies, so each runs in its own Pixi environment. Conda is not part of the supported installation workflow for this repository.

## Quick start

Install all locked environments from the repository root:

```bash
pixi install
```

Open Cheese3D with the environment matching the project's configured backend:

```bash
# DLC3
pixi run -e dlc cheese3d --path /data/disk2/home/tony interactive

# Lightning Pose on the CUDA 12 stack
pixi run -e lp cheese3d --path /data/disk2/home/tony interactive

# Experimental Lightning Pose CUDA 13 stack
pixi run -e lp-cu13 cheese3d --path /data/disk2/home/tony interactive

# SLEAP
pixi run -e sleap cheese3d --path /data/disk2/home/tony interactive
```

The `--path` value is the directory containing project folders such as `demo1` and `demo2`; it is not the project folder itself.

For an interactive shell, use `pixi shell -e dlc`, `pixi shell -e lp`, or `pixi shell -e sleap`. Do not enter one backend shell from inside another backend shell because inherited CUDA library paths can mix incompatible cuDNN versions.

## Pixi environments

| Environment | Purpose | Important pinned packages |
|---|---|---|
| `dlc` | DLC3 GUI, training, and inference | DeepLabCut 3.0.1, PyTorch 2.10.0, pytest/textual-dev/py-spy |
| `lp` | Preferred Lightning Pose CUDA 12 environment | Lightning Pose extra, PyTorch 2.7.1 cu126, JAX 0.4.36 CUDA 12, pytest/textual-dev/py-spy |
| `lp-cu13` | Experimental Lightning Pose CUDA 13 environment | Lightning Pose extra, CPU JAX 0.4.36, pytest/textual-dev/py-spy |
| `sleap` | Isolated SLEAP training and inference | SLEAP 1.6.1, sleap-io 0.6.4, sleap-nn 0.1.0, PyTorch 2.7.1 cu128, pytest/textual-dev/py-spy |
| `triangulation-gpu` | Isolated JAX GPU triangulation worker (subprocess only, launched automatically by `Ch3DProject.triangulate`) | JAX 0.4.36 CUDA 12; no pose backend, PyTorch, or dev tooling |

The CUDA 13 LP environment exists to isolate an experimental newer CUDA/PyTorch stack. The CUDA 12 `lp` environment is the normal choice.

There is no `default`/`dev-*` environment: dev tooling (pytest, textual-dev, py-spy) is folded directly into each backend environment above, and `cheese3d` is editable-installed everywhere, so a source change is live in every environment the moment it's saved -- there was never a separate place "the changes" lived. Every command below passes `-e <name>` explicitly; there is no supported bare `pixi run`/`pixi shell` without one.

## Project configuration layout

The intended canonical layout places user-editable configuration files directly under the Cheese3D project directory:

```text
project/
├── config.yaml                         # Cheese3D project/session/model config
├── dlc_backend_config.yaml             # canonical DLC project config, when DLC is active
├── dlc_network_config.yaml             # latest generated DLC3 network-config snapshot
├── lightning_pose_network_config.yaml  # canonical LP training/network config
├── sleap_network_config.yaml           # canonical SLEAP-NN training/network config
└── model/
    └── MODEL_NAME/
        └── backend/
            ├── config.yaml             # backend-native synchronized working copy
            ├── labels/data/checkpoints # backend-owned artifacts
            └── ...
```

Current synchronization behavior:

- `project/config.yaml` remains Cheese3D's primary project config.
- On backend construction, Cheese3D supplies a stable root-level canonical path based on `backend_type`.
- DLC, Lightning Pose, and SLEAP pull the root canonical file into the backend-required location before native operations.
- Validated or GUI-updated backend settings are copied back to the root canonical file.
- DLC generates `pytorch_config.yaml` inside its model/shuffle folders. After successful training/evaluation, the newest generated file is published to `project/dlc_network_config.yaml` as an inspection snapshot. DLC still owns the nested generated copy.
- A missing root canonical file is bootstrapped from the existing backend-local config, preserving older projects.

The synchronization implementation is currently in:

- `packages/cheese3d/cheese3d/backends/core.py`
- `packages/cheese3d/cheese3d/project.py`
- `packages/cheese3d/cheese3d/backends/dlc.py`
- `packages/cheese3d/cheese3d/backends/lightning_pose.py`
- `packages/cheese3d/cheese3d/backends/sleap.py`

### Config synchronization status

- [x] Root canonical filenames are assigned while building DLC, LP, and SLEAP backends.
- [x] Shared pull/push helpers copy between root and backend-native paths.
- [x] DLC pulls before initialization/training and pushes project settings afterward.
- [x] DLC publishes the newest generated network YAML after training.
- [x] Lightning Pose pulls before loading/training and pushes config/keypoint/GUI updates.
- [x] SLEAP pulls before loading/training and pushes GUI/training updates.
- [ ] Run non-GPU unit tests and construction tests for this newest synchronization patch.
- [ ] Open existing `demo1` and `demo2` once in their correct environments to bootstrap the root files.
- [ ] Verify that editing each root YAML while the GUI is open is picked up on the next train action.
- [ ] Add automated round-trip tests for root-to-backend and backend-to-root synchronization.
- [ ] Decide whether `dlc_network_config.yaml` should remain read-only documentation or become an input used to seed the next DLC shuffle. It is currently an output snapshot.

## Shared Cheese3D and GUI work

The current tree includes the following shared workflow improvements:

- Backend selection supports `dlc`, `lightning_pose`, and `sleap`.
- Training is exposed separately from model creation.
- Backend-specific training controls are shown in the GUI.
- GPU IDs can be selected; multi-camera inference partitions videos across selected GPUs.
- Checkpoints can be listed and selected with available validation metrics.
- Tracking exposes batch/GPU controls and per-camera progress reporting.
- Training can be stopped early from the GUI.
- GUI completion handling was adjusted for training, inference, labeling, video generation, and visualization so long operations do not permanently leave the interface in a busy state.
- Directory navigation supports leaving the Cheese3D directory and moving to the parent directory using physical controls.
- Visualization exposes confidence thresholds and playback frame-rate controls.
- Video rendering uses parallel CPU workers with an adjustable worker count.
- Napari visualization includes caching/precomputation work, smaller markers, memory limits, and optional CUDA-related acceleration paths.
- Triangulation has an optional isolated JAX GPU worker environment.

These features have accumulated across many patches. Treat end-to-end GUI behavior as requiring regression testing even where focused tests previously passed.

## DLC3 backend

Implemented:

- DLC legacy config keys are removed and the project is validated as DLC3.
- Compatible labels are synchronized into DLC's `labeled-data` directory.
- Training/test fraction is adjustable and a new training set is created when it changes.
- The created shuffle is propagated into training, evaluation, checkpoint selection, and tracking instead of assuming shuffle 1.
- DLC3 architectures exposed by the installed release can be selected.
- Single-animal DLCRNet receives PAF edges derived from the Cheese3D skeleton.
- Batch size, learning rate, epochs, augmentation, save/validation interval, and snapshots-to-keep are configurable.
- Multi-GPU training uses DLC's supported runner GPU list; inference distributes different camera videos between GPU workers.
- Per-camera inference progress is reported.

Known risks / remaining work:

- [ ] Regression-test all DLC architectures exposed in the GUI; multi-GPU behavior is architecture/framework dependent.
- [ ] Test the new root config synchronization against multiple existing shuffles and iterations.
- [ ] Confirm the latest generated `pytorch_config.yaml` selection is correct when two runs finish very close together.
- [ ] Ensure tracking defaults always resolve the selected training fraction and shuffle after restarting Cheese3D.
- [ ] Investigate any remaining flat-loss behavior independently from the GUI; batch size 128 previously exhausted a 48 GiB GPU and is not a safe default.

## Lightning Pose backend

Implemented:

- DLC labels can be converted into Lightning Pose's CSV/image layout.
- Cheese3D annotation labels can be merged into the LP dataset.
- Training uses Lightning Pose's public training path and refreshes the inference model afterward.
- Common training controls and LP-specific augmentation controls are exposed.
- Multiple backbones are selectable; ViT inputs are automatically made square.
- Scheduler milestones are scaled when GUI `max_epochs` is reduced.
- Multi-GPU DDP is configured from comma-separated GPU IDs.
- ViT multi-GPU training enables unused-parameter detection where required.
- Checkpoint selection and validation metrics are exposed.
- Multi-camera inference is partitioned across GPU workers and converted to DLC-compatible HDF5 for triangulation.
- Progress JSON writes were hardened against concurrent DDP ranks.

Previously validated before the current config patch:

- A short single-GPU training/inference smoke test completed.
- A two-GPU DDP training smoke test launched ranks 0 and 1.
- Two-camera inference assigned one worker to each physical GPU.

Known risks / remaining work:

- [ ] Do not run GPU validation until the machine owner confirms the GPUs are available.
- [ ] Re-test 10 epochs in both `lp` and `lp-cu13` after GPU access is restored.
- [ ] Confirm both GPUs sustain useful utilization for each supported backbone; DDP startup alone does not guarantee balanced work.
- [ ] Verify epoch/loss/validation progress remains visible while suppressing noisy rank/NCCL output.
- [ ] Diagnose remaining NCCL TCPStore broken-pipe messages. They generally indicate one rank exited early and are secondary to the first rank's exception.
- [ ] Verify GUI refresh/completion behavior after LP inference and visualization in a real browser session.
- [ ] Update the older `LIGHTNING_POSE_IMPLEMENTATION_STATUS.md`; it describes an early prototype and is now substantially stale.

## SLEAP backend

Implemented:

- SLEAP is registered as a Cheese3D backend.
- Its dependency stack is isolated in the `sleap` Pixi environment.
- DLC-style labels and Cheese3D annotations are converted into a portable `labels.slp` package.
- The current workflow uses SLEAP-NN's single-instance training pipeline.
- GUI controls include backbone, epochs, train/validation batch sizes, validation fraction, optimizer, learning rate, steps per epoch, early stopping, checkpoint retention, and geometric augmentation.
- SLEAP checkpoints and validation metrics can be listed and selected.
- Inference calls SLEAP-NN tracking and converts predictions into DLC-compatible HDF5.
- Camera videos are partitioned across selected GPUs for inference.
- Multi-GPU training config uses DDP and preserves the selected physical GPU indices.

Previously validated before the current config patch:

- Single-GPU SLEAP training and inference smoke tests completed.
- Two-GPU DDP training launched both ranks.
- Two-camera inference assigned workers to separate physical GPUs.

Known limitations / remaining work:

- [ ] Do not run further GPU tests until GPU use is explicitly reauthorized.
- [ ] SLEAP support is currently single-instance; top-down/bottom-up multi-instance workflows are not implemented.
- [ ] Importing an arbitrary existing SLEAP project is not implemented.
- [ ] SLEAP native label edits are not exported back into Cheese3D; Cheese3D annotations are currently treated as authoritative.
- [ ] Automatic frame extraction through SLEAP is not implemented; use Cheese3D's existing manual workflow.
- [ ] Validate resume/interruption behavior and checkpoint selection after process restart.
- [ ] Add focused unit tests for config generation, label conversion, checkpoint metadata, and root-config synchronization.

## Triangulation and visualization

The repeated Numba warning from Aniposelib around `np.diff(..., axis=0)` is a Numba typing failure inside Aniposelib's bundle-adjustment/temporal optimization function. Numba falls back to object mode; it is noisy and slower but is not, by itself, evidence that RANSAC is running or that triangulation failed.

Potential follow-up:

- [ ] Pin a compatible Numba/Aniposelib pair or patch the affected function to avoid unsupported keyword typing.
- [ ] Benchmark the isolated JAX linear-triangulation worker against CPU triangulation.
- [ ] Keep nonlinear bundle adjustment and temporal constraints on CPU unless their numerical equivalence is verified.
- [ ] Add a clear terminal line stating which triangulation implementation/device is active.
- [ ] Validate 2-D coordinate scaling at every stage using the actual decoded video resolution and calibration resolution.

The earlier consistent 3-D reprojection offset was associated with mismatched coordinate/video resolution handling even when reprojection error appeared low. Any future format conversion must preserve the coordinate system used for calibration and triangulation.

## Testing and safety notes

Dev tooling is part of each backend environment now (no separate `dev-*` environments); run tests directly:

```bash
pixi run -e dlc pytest packages/cheese3d/tests
pixi run -e lp pytest packages/cheese3d/tests
pixi run -e sleap pytest packages/cheese3d/tests
```

Recommended immediate CPU-only checks:

```bash
python -m compileall packages/cheese3d/cheese3d
pixi run -e dlc pytest packages/cheese3d/tests
```

Do not run GPU smoke tests, training, inference, `nvidia-smi` monitoring loops, or commands that initialize CUDA until the machine owner says GPU validation may resume.

Large videos and label datasets live outside the repository under `/data/disk2/home/tony`. Do not copy them into Git. Preserve existing user edits and inspect processes before terminating anything.

## Recommended next steps

1. Run compile and CPU-only unit tests for the current root-config patch.
2. Add isolated filesystem tests proving each canonical root config bootstraps, pulls, and pushes correctly.
3. Open `demo1` in `dlc` and `demo2` in `lp` when permitted, then confirm the expected root-level YAML files are created without altering labels or checkpoints.
4. Add a small SLEAP demo project and verify `sleap_network_config.yaml` is created at its root.
5. Update or replace the stale Lightning Pose status document.
6. Commit the config synchronization separately from unrelated GUI/backend work if the current changes can be cleanly separated.
7. After GPU use is reauthorized, run one controlled 10-epoch test per backend/environment and record exact model, batch size, GPU IDs, utilization, checkpoint, and validation metric.

## Files most likely to be edited next

- `pixi.toml` — environment definitions and dependency pins
- `packages/cheese3d/cheese3d/project.py` — backend construction and root config routing
- `packages/cheese3d/cheese3d/backends/core.py` — common backend/config/progress utilities
- `packages/cheese3d/cheese3d/backends/dlc.py` — DLC3 config, datasets, training, and inference
- `packages/cheese3d/cheese3d/backends/lightning_pose.py` — LP conversion, training, DDP, and inference
- `packages/cheese3d/cheese3d/backends/sleap.py` — SLEAP config, labels, training, and inference
- `packages/cheese3d/cheese3d/interactive.py` — GUI controls and completion handling
- `packages/cheese3d/cheese3d/visualization.py` — Napari caching/playback/overlay performance
- `packages/cheese3d/tests/` — regression and backend tests

## Handoff checklist

- [ ] Read this document and inspect `git diff` before editing.
- [ ] Confirm which Pixi environment matches the target project's backend.
- [ ] Keep backend dependency stacks isolated.
- [ ] Keep user-editable configs in the project root and backend-native copies synchronized.
- [ ] Preserve large datasets outside Git.
- [ ] Avoid GPU work until authorization is restored.
- [ ] Record commands, environment, checkpoint, and validation result for every future smoke test.
- [ ] Cross off completed items in this file as work progresses.
