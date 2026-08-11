# Lightning Pose Integration Status and Implementation Checklist

This document tracks the work required to make Lightning Pose (LP) a complete, reliable alternative to DeepLabCut (DLC) in Cheese3D.

Last reviewed: 2026-08-10  
Reviewed branch/commit: `tf_patch` at `6b00842`  
Primary implementation: `packages/cheese3d/cheese3d/backends/lightning_pose.py`

## Status summary

Current maturity: **prototype / inference-only partial integration**.

The repository contains a registered Lightning Pose backend, LP dependencies, video preprocessing, prediction CSV parsing, DLC-compatible HDF5 export, and a single-view tracking path. However, a normal user cannot currently create or import a Lightning Pose model through Cheese3D, and the frame extraction, label exchange, training, and multiview workflows are not implemented.

The integration should not yet be presented as a drop-in DLC replacement.

## Definition of done

Lightning Pose can be considered a supported DLC alternative when all of the following are true:

- [ ] A user can select Lightning Pose in both the interactive UI and configuration file.
- [ ] A user can create a new LP-backed Cheese3D project without manually arranging backend files.
- [ ] A user can import an existing LP model/project.
- [ ] A user can extract frames, annotate them, exchange labels with Cheese3D, and train a model—or the product explicitly supports and documents an inference-only workflow.
- [ ] Single-view and intended multiview inference modes work predictably.
- [ ] LP predictions are converted into the exact HDF5 layout expected by Anipose.
- [ ] Multiple recordings and cameras cannot overwrite one another's videos or predictions.
- [ ] Track-to-triangulate works end to end on representative data.
- [ ] Automated tests exercise construction, inference, conversion, caching, errors, and integration.
- [ ] Installation and user documentation accurately describe the LP workflow and limitations.

## What is already implemented

- [x] `lightning_pose` is registered as a built-in pose backend.
- [x] Backend-specific imports defer importing Lightning Pose until it is needed.
- [x] DLC and Lightning Pose installations are treated as mutually exclusive.
- [x] The package declares an LP dependency set using Lightning Pose 2.2.0.
- [x] Pixi declares LP environments for CUDA configurations.
- [x] `ffprobe` checks whether a video is MP4/H.264/YUV420P.
- [x] `ffmpeg` converts incompatible videos to LP-compatible input.
- [x] LP CSV predictions are parsed and metadata/non-2D columns are filtered.
- [x] LP predictions can be written as DLC-style HDF5 files.
- [x] The backend can call LP inference one video at a time.
- [x] Existing final HDF5 outputs are skipped.
- [x] Basic helper tests exist for CSV parsing and video preprocessing.
- [x] EKS infrastructure can refer to Lightning Pose primitive models.

---

## P0 — Backend creation and loading are not functional

### P0.1 New LP projects expect a configuration that does not exist

Current behavior:

1. `build_model_backend()` constructs `LightningPoseBackend` under `model/<name>/backend`.
2. The constructor creates that directory.
3. `_update_config()` immediately reads `backend/config.yaml` through `ModelConfig.from_yaml_file()`.
4. Cheese3D has not created or copied an LP configuration into that directory.
5. Construction therefore fails for a normal newly created project.

Relevant code:

- `packages/cheese3d/cheese3d/project.py`, `build_model_backend()`
- `packages/cheese3d/cheese3d/backends/lightning_pose.py`, `__init__()` and `_update_config()`

Tasks:

- [ ] Decide what "create an LP model" means in Cheese3D.
- [ ] Identify the supported Lightning Pose API for creating a project/configuration.
- [ ] Generate a valid LP project layout before calling `Model.from_dir()`.
- [ ] Populate LP keypoint names from `KeypointConfig`.
- [ ] Populate videos and view names from Cheese3D session/view configuration.
- [ ] Translate Cheese3D crop settings into LP configuration or document why crops are applied elsewhere.
- [ ] Validate required LP configuration keys before mutating the filesystem.
- [ ] Produce a clear error if initialization cannot be completed.
- [ ] Avoid partially initialized backend directories after failure.

Acceptance criteria:

- [ ] A config with `model.backend_type: lightning_pose` and a new model name loads successfully.
- [ ] The resulting backend directory is a valid LP project loadable by Lightning Pose itself.
- [ ] Reloading the Cheese3D project is idempotent and does not recreate or corrupt the model.
- [ ] A unit/integration test covers first creation and subsequent reload.

### P0.2 Importing an existing LP project is unimplemented

Current behavior:

- `LightningPoseBackend.from_existing()` raises `NotImplementedError`.
- The generic import path is hard-coded to parse and load a DLC folder, so it cannot dispatch to LP even if `from_existing()` is implemented.

Tasks:

- [ ] Extend model import configuration so the backend type is known before import.
- [ ] Remove DLC-specific folder parsing from the generic `build_model_backend()` import branch.
- [ ] Implement validation for an existing LP project/model directory.
- [ ] Decide whether import copies, symlinks, or references the source model.
- [ ] Preserve or rewrite internal paths safely when copying.
- [ ] Determine the imported model name without relying on DLC folder naming conventions.
- [ ] Store enough backend configuration for reliable project reloads.
- [ ] Handle missing config, checkpoints, or incompatible LP versions with actionable errors.

Acceptance criteria:

- [ ] An existing trained LP project can be selected and imported.
- [ ] The imported model can run inference after the original source directory is moved, if import semantics are "copy."
- [ ] Reloading the Cheese3D project selects LP rather than DLC.
- [ ] Tests cover valid import, invalid directory, missing checkpoint, and name/path handling.

### P0.3 The interactive project wizard always creates DLC configuration

Current behavior:

- The model wizard offers only `create` or `import`.
- It has no backend selector.
- The create path constructs `ModelConfig(model_name)`, retaining the default `backend_type="dlc"`.
- The import path assumes DLC-specific fields such as `experimenter` and `date`.

Tasks:

- [ ] Add a backend selector to the model wizard.
- [ ] Show backend-specific fields only when applicable.
- [ ] Pass `backend_type` into `ModelConfig` on creation.
- [ ] Pass the selected backend type through the import path.
- [ ] Stop reading DLC-only attributes for LP imports.
- [ ] Validate that the required optional dependency is installed before completing the wizard.
- [ ] Display a useful environment-installation message if LP is unavailable.

Acceptance criteria:

- [ ] The UI can create an LP configuration without editing YAML manually.
- [ ] The UI can import an LP model.
- [ ] DLC creation/import continues to work unchanged.
- [ ] UI tests or model-wizard logic tests cover both backends.

---

## P0 — Cheese3D labeling and training lifecycle is unimplemented

The following methods currently raise `NotImplementedError`:

- `import_c3d_labels()`
- `export_c3d_labels()`
- `extract_frames()`
- `train()`

This affects more than the four direct operations. `Ch3DProject.extract_frames()` and `Ch3DProject.train()` automatically call label import/export hooks, so the shared manual frame picker and annotator paths also fail with LP.

### P0.4 Define the supported product scope

Choose one scope before implementing details:

- [ ] **Full lifecycle:** Cheese3D creates, labels, trains, and runs LP models.
- [ ] **Inference-only:** Users train externally and Cheese3D imports/runs trained LP models.

If inference-only is selected:

- [ ] Disable unsupported UI and CLI actions for LP instead of allowing raw `NotImplementedError` exceptions.
- [ ] Document the external LP training workflow.
- [ ] Explain how LP keypoints, views, crops, and video names must correspond to Cheese3D.
- [ ] Add explicit capability flags to the backend interface if needed.

If full lifecycle is selected, complete the following sections.

### P0.5 Label interchange

Tasks:

- [ ] Document the Cheese3D label directory and table schema.
- [ ] Document the LP label/image directory and table schema for version 2.2.0.
- [ ] Define deterministic mapping for image paths, scorers, keypoints, views, and missing values.
- [ ] Implement Cheese3D-to-LP label import.
- [ ] Implement LP-to-Cheese3D label export.
- [ ] Preserve labels through a round trip without coordinate or image reassignment.
- [ ] Handle empty label sets and partially labeled frames.
- [ ] Detect keypoint-name mismatches and duplicate labels.
- [ ] Ensure paths remain valid if a project is moved.

Acceptance criteria:

- [ ] A labeled Cheese3D frame appears at the correct coordinates in LP.
- [ ] An LP-edited label reappears correctly in the Cheese3D annotator.
- [ ] Round-trip tests cover multiple cameras, missing points, and duplicate frame stems.

### P0.6 Frame extraction

Tasks:

- [ ] Select an LP API or Cheese3D-owned strategy for automatic extraction.
- [ ] Respect a caller-provided video subset.
- [ ] Ensure extracted frames retain recording and camera identity.
- [ ] Prevent collisions when different recordings have the same video stem.
- [ ] Integrate automatically extracted frames with the shared Cheese3D annotator.
- [ ] Make repeated extraction idempotent or explicitly warn before duplication.

Acceptance criteria:

- [ ] Automatic extraction completes for a multi-camera recording.
- [ ] Manual extraction through the existing Napari picker works for LP.
- [ ] Extracted frames can be labeled and then used for LP training.

### P0.7 Training

Tasks:

- [ ] Select the supported LP training API.
- [ ] Build/refresh LP training data from current labels.
- [ ] Map the `gpu` argument to the LP/PyTorch device configuration.
- [ ] Define the meaning of `iterate_dataset` for LP.
- [ ] Surface progress and errors in CLI and Textual UI.
- [ ] Record the chosen checkpoint and make reload deterministic.
- [ ] Decide how interrupted training resumes.
- [ ] Validate that at least one usable labeled dataset exists.

Acceptance criteria:

- [ ] A small test dataset can train far enough to produce a loadable checkpoint.
- [ ] The trained checkpoint is selected for inference after project reload.
- [ ] CPU/no-GPU errors and invalid GPU IDs produce actionable messages.

---

## P1 — Multiview inference is disabled

Current behavior:

- The multiview branch in `LightningPoseBackend.track()` is commented out.
- Every camera video is sent separately to `predict_on_video_file()`.
- A comment notes a mismatch between the model configuration and an externally described single-view-transformer setup; this uncertainty is unresolved in code or documentation.

Tasks:

- [ ] Determine which LP model modes Cheese3D officially supports: single-view, multiview, single-view transformer, or a defined subset.
- [ ] Confirm the correct LP 2.2.0 inference API for each supported mode.
- [ ] Read model view order from configuration.
- [ ] Match Cheese3D view keys directly rather than by filename substring.
- [ ] Validate missing, extra, or reordered views before inference.
- [ ] Restore multiview inference using public APIs where possible.
- [ ] Avoid calling private `self.model._load()` unless no public alternative exists.
- [ ] Define behavior when some camera outputs already exist.
- [ ] Validate synchronized frame counts where the LP mode requires them.

Acceptance criteria:

- [ ] A multiview model receives all camera videos in configured view order.
- [ ] Incorrect view sets fail before expensive inference begins.
- [ ] Single-view models still process each camera correctly.
- [ ] Tests cover reordered dictionaries, missing views, extra views, and partial cached output.

---

## P1 — Prediction paths and cache behavior are fragile

### P1.1 Filename collisions

Current behavior:

- Preprocessed outputs use only `<video.stem>.mp4`.
- LP internal predictions are located using only `<video.stem>.csv`.
- Final Anipose inputs use only `<video.stem>.h5`.

Two recordings or views with the same stem can overwrite or incorrectly reuse one another's artifacts.

Tasks:

- [ ] Define a stable artifact identity containing recording/session and view.
- [ ] Preserve extensions safely when source formats vary.
- [ ] Place preprocessing and prediction artifacts in recording-scoped directories or use collision-proof names.
- [ ] Store source-path/fingerprint metadata with cached artifacts.
- [ ] Detect legacy ambiguous caches and regenerate or report them.

Acceptance criteria:

- [ ] Two source files with identical basenames generate distinct artifacts.
- [ ] Re-running the same source reuses the correct artifact.
- [ ] Moving between sessions cannot silently reuse predictions from another session.

### P1.2 Cache invalidation is incomplete

Current behavior:

- Preprocessed video reuse checks source modification time and output format.
- Final HDF5 reuse checks only whether the file exists.
- LP CSV reuse checks only whether the file exists.
- Model checkpoint changes, LP config changes, and source video changes do not invalidate predictions.

Tasks:

- [ ] Define a prediction cache fingerprint: source video identity, model checkpoint, relevant config, and code/schema version.
- [ ] Invalidate CSV/HDF5 outputs when the source or model changes.
- [ ] Provide a documented force/recompute option.
- [ ] Write outputs atomically so interruptions do not leave apparently valid files.
- [ ] Validate cached HDF5 schema before skipping work.

Acceptance criteria:

- [ ] Changing the selected model checkpoint causes recomputation.
- [ ] Changing the source video causes recomputation.
- [ ] Interrupted conversion does not leave a cache hit on the next run.

### P1.3 Output existence and errors are not validated

Tasks:

- [ ] After LP inference, verify that the expected CSV exists.
- [ ] Include source video and expected output path in missing-output errors.
- [ ] Validate prediction row count against the source/preprocessed video frame count.
- [ ] Validate required `x`, `y`, and `likelihood` columns for every expected keypoint.
- [ ] Handle empty and malformed CSV files explicitly.
- [ ] Clean up incomplete temporary files after failure.

Acceptance criteria:

- [ ] Missing or malformed LP output raises a concise, actionable exception.
- [ ] No incomplete HDF5 output is treated as successful.

---

## P1 — Coordinate and video preprocessing correctness needs validation

Current preprocessing:

- Converts to H.264/YUV420P MP4.
- Uses all-intra frames (`-g 1`).
- Sets sample aspect ratio to 1.
- Drops audio.

Open questions:

- Does LP inference return coordinates in the original frame geometry?
- Are Cheese3D camera crops expected to be applied before inference?
- Does `setsar=1` change only display metadata or reveal existing non-square-pixel assumptions?
- Does preprocessing preserve frame count, frame order, timestamps, and dimensions for all supported input formats?

Tasks:

- [ ] Document the coordinate space expected by Anipose and Cheese3D.
- [ ] Decide where crop transformations are applied.
- [ ] Apply inverse crop/resize transforms before HDF5 export if required.
- [ ] Compare source and converted frame count/dimensions.
- [ ] Test odd dimensions, variable frame rate, grayscale, and corrupted videos.
- [ ] Decide whether audio removal matters for downstream synchronization and document it.
- [ ] Capture enough `ffmpeg` stderr in raised errors to diagnose conversion failure.

Acceptance criteria:

- [ ] A known pixel coordinate survives preprocessing/inference/export in the correct camera coordinate system.
- [ ] Frame indices correspond exactly to the frames used by triangulation.
- [ ] Crop settings either work correctly or are rejected/documented.

---

## P1 — DLC-compatible HDF5 conversion needs stronger guarantees

Current behavior:

- Reads a three-row CSV header.
- Filters to `x`, `y`, and `likelihood`.
- Optionally replaces the scorer level.
- Writes `df_with_missing` using pandas/PyTables.
- Does not overwrite an existing HDF5 file.

Tasks:

- [ ] Confirm actual LP 2.2.0 CSV schemas for every supported model mode.
- [ ] Confirm Anipose's required index, column names, dtypes, HDF key, and table/fixed format.
- [ ] Verify how LP represents confidence and whether it is semantically equivalent to DLC likelihood.
- [ ] Preserve or deliberately normalize frame indices.
- [ ] Validate expected keypoints against Cheese3D configuration.
- [ ] Define behavior for additional LP metrics/columns.
- [ ] Avoid applying the scorer rewrite twice in `lp_csv_to_dlc_h5()`/`dlc_df_to_h5()`.
- [ ] Add a safe overwrite/atomic-update path for stale artifacts.

Acceptance criteria:

- [ ] Anipose reads generated HDF5 without DLC installed.
- [ ] Known CSV values round-trip exactly into expected HDF5 columns.
- [ ] Missing keypoints and malformed headers fail clearly.
- [ ] A real LP prediction file is covered by a fixture-based test.

---

## P1 — Dependencies and environments are inconsistent

Current issues:

- The actual optional dependency is named `lp`.
- Runtime error text tells users to install `cheese3d[lightning-pose]`, which does not exist.
- `test_plugins.py` expects an extra named `lightning-pose`, an older EKS dependency location/version, and an older conflict structure.
- The checked Pixi environments were incomplete during the 2026-08-10 review: the development environment lacked pytest, and the LP development environment lacked both pytest and `lightning_pose`.
- Pixi could not refresh under the review sandbox because its global cache had no writable layer.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set for LP environments and should be justified because it changes checkpoint-loading security behavior.

Tasks:

- [ ] Choose one public extra name: `lp` or `lightning-pose`.
- [ ] Update runtime messages, documentation, Pixi feature names, and tests consistently.
- [ ] Update stale dependency tests to reflect the intended package layout.
- [ ] Verify both `dev-lp` and release installation from a clean environment.
- [ ] Add a CPU-capable LP test environment if feasible.
- [ ] Document supported Python, CUDA, PyTorch, JAX, LP, and EKS versions.
- [ ] Document why DLC and LP must be mutually exclusive, or remove the restriction if no longer technically necessary.
- [ ] Review and document the checkpoint trust/security implications of `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- [ ] Ensure CI resolves the lockfile and imports LP.

Acceptance criteria:

- [ ] The documented install command succeeds in a clean environment.
- [ ] `python -c "import cheese3d; import lightning_pose"` succeeds there.
- [ ] The targeted LP test suite runs in CI.
- [ ] Dependency metadata tests match the actual intended extras.

---

## P1 — Automated testing is insufficient

Existing tests cover only:

- Filtering LP CSV columns.
- Recognizing compatible video metadata.
- Building the expected `ffmpeg` command.
- Registry presence indirectly through plugin tests.

### Required unit tests

- [ ] New backend creation.
- [ ] Existing backend reload.
- [ ] Existing LP model import.
- [ ] Configuration override merge.
- [ ] Missing and malformed LP config.
- [ ] Checkpoint discovery/selection.
- [ ] Single-view mocked inference.
- [ ] Multiview mocked inference.
- [ ] View ordering and validation.
- [ ] Prediction naming with duplicate stems.
- [ ] Cache hit and invalidation behavior.
- [ ] Missing/malformed prediction CSV.
- [ ] CSV-to-HDF5 schema and numeric values.
- [ ] Label import/export round trip, if supported.
- [ ] Frame extraction and training dispatch, if supported.
- [ ] Missing `ffmpeg` and failed `ffprobe`/`ffmpeg` behavior.
- [ ] Paths containing spaces and unusual characters.

### Required integration tests

- [ ] Load a minimal real LP project fixture.
- [ ] Infer on a tiny video and produce CSV/HDF5.
- [ ] Have Anipose load the generated HDF5.
- [ ] Track all views of one session.
- [ ] Triangulate a minimal calibrated session.
- [ ] Reload the Cheese3D project and reuse valid artifacts.
- [ ] Train a tiny model/checkpoint, if full lifecycle is supported.

### CI matrix

- [ ] Core environment without DLC or LP.
- [ ] DLC environment.
- [ ] LP environment.
- [ ] At least one supported Linux/CUDA combination where GPU testing is available.
- [ ] CPU-only helper and mocked-backend tests on every pull request.

---

## P2 — Documentation is DLC-only

Current documentation problems:

- The README describes Cheese3D as built on DLC and Anipose.
- The configuration reference lists only `Literal["dlc"]`.
- The CLI import help says only DLC is valid.
- Installation and quick-start guides contain no LP setup path.
- Project-layout documentation calls the backend directory a DLC project.
- The UI screenshot and instructions show only DLC model selection.

Tasks:

- [ ] Update the root and package READMEs once support reaches the intended maturity.
- [ ] List all supported `backend_type` values and exact spelling.
- [ ] Document LP-specific `backend_options` with examples.
- [ ] Add LP installation instructions for supported platforms/CUDA versions.
- [ ] Add an LP create/import quick start.
- [ ] Document external-training steps if the backend remains inference-only.
- [ ] Explain single-view versus multiview model requirements.
- [ ] Explain video preprocessing and artifact locations.
- [ ] Explain cache invalidation and force-recompute behavior.
- [ ] Add LP troubleshooting for missing checkpoints, views, codecs, and prediction files.
- [ ] Update CLI help and project-layout descriptions to be backend-neutral.

Acceptance criteria:

- [ ] A new user can install, configure, and run the supported LP workflow using documentation alone.
- [ ] No user-facing page incorrectly claims an LP feature is implemented.

---

## P2 — Backend API and maintainability improvements

Tasks:

- [ ] Add explicit backend capabilities such as `can_train`, `can_extract_frames`, and `can_manage_labels` if different backends intentionally support different lifecycles.
- [ ] Replace raw `NotImplementedError` exposure with user-facing capability messages.
- [ ] Add docstrings and return types to LP helper functions and methods.
- [ ] Replace `print()` with the project's consistent progress/reporting mechanism.
- [ ] Remove unused `predict_video` import when multiview is disabled, or use it in the restored implementation.
- [ ] Remove obsolete commented-out inference code after the final mode design is implemented.
- [ ] Separate project management, inference, conversion, and video preprocessing into testable units if the backend grows.
- [ ] Avoid private Lightning Pose APIs where possible.
- [ ] Pin or guard against API changes if version upgrades are expected.

Acceptance criteria:

- [ ] UI and CLI can determine supported operations without triggering exceptions.
- [ ] Backend code contains no large disabled implementation blocks.
- [ ] Public errors identify the model, recording/view, and corrective action.

---

## Suggested implementation sequence

### Milestone 1 — Reliable inference using externally trained LP models

- [ ] Resolve extra/environment naming and restore runnable LP tests.
- [ ] Implement LP model import/loading.
- [ ] Add backend selection to YAML, CLI, and UI import flow.
- [ ] Harden single-view inference and artifact naming.
- [ ] Validate CSV-to-HDF5 output with Anipose.
- [ ] Add an end-to-end track/triangulate test.
- [ ] Document the inference-only workflow and limitations.

Deliverable: users can import a trained LP model and reliably run Cheese3D tracking and triangulation.

### Milestone 2 — Multiview support

- [ ] Decide supported LP multiview modes.
- [ ] Restore view-aware multiview inference.
- [ ] Add view/frame validation and multiview fixtures.
- [ ] Document model/view requirements.

Deliverable: Cheese3D can use the intended LP multiview architecture safely.

### Milestone 3 — Native project creation and training

- [ ] Create valid LP projects from Cheese3D configuration.
- [ ] Implement label interchange.
- [ ] Implement automatic/manual frame extraction integration.
- [ ] Implement training and checkpoint selection.
- [ ] Add lifecycle tests and a complete tutorial.

Deliverable: Lightning Pose provides the same user-visible model lifecycle that DLC currently provides.

### Milestone 4 — Release readiness

- [ ] Run the full DLC, LP, EKS, synchronization, and triangulation test suites.
- [ ] Test clean installations on supported platforms.
- [ ] Complete documentation and migration notes.
- [ ] Benchmark prediction quality, speed, and resource use on representative Cheese3D recordings.
- [ ] Remove the experimental/prototype label only after all release-blocking items pass.

---

## Progress log

Add dated entries here when completing or changing an item. Include the commit or pull request and the verification performed.

| Date | Item | Change | Verification | Commit/PR |
|---|---|---|---|---|
| 2026-08-10 | Initial audit | Created detailed implementation checklist | Static repository review; test execution blocked by incomplete Pixi environments/cache permissions | `6b00842` |

## Review notes

- The worktree contained an existing modification to `pixi.lock` during this audit. It was not evaluated as an intentional LP implementation change and should be reviewed before committing.
- This checklist describes repository state at the commit listed above. Re-run the focused audit after dependency or Lightning Pose version upgrades because LP APIs and output schemas may change.
