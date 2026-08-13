# SLEAP Backend — Problems and Handoff

**Status:** DeepLabCut and Lightning Pose run the full pipeline end to end. SLEAP trains but
cannot be made to fit in GPU memory at any batch size tried, and the one run that did complete
produced an unusable model.

Everything below is measured, not inferred.

| | Date | Environment |
|---|---|---|
| | 2026-08-12 | SLEAP 1.6.1 · sleap-nn 0.1.0 · sleap-io 0.6.4 · torch 2.7.1+cu128 · 2 × RTX A6000 (47.4 GiB each) |

**Dataset:** 958 labeled images, 28 keypoints, 640×512, six camera views, one test session
(`20231031_chew`), seeded from the DLC project in `/data/disk2/home/tony/cheese3d2_test_set`.

## Result summary

| Backend | Network | Result | Training time | Model quality |
|---|---|---|---|---|
| DeepLabCut | `resnet_50` | **PASSED** | 25.7 min | RMSE 4.18 px, mAP 100 |
| Lightning Pose | `resnet50_animal_ap10k` | **PASSED** | 15.6 min | val loss 0.00395 |
| SLEAP | `convnext_tiny` | **BLOCKED** | — | CUDA OOM at every batch tried |

Both passing backends produced 3D triangulation and labeled overlay videos from the same dataset
and the same six-camera session. The dataset, label conversion, calibration and triangulation
stages are therefore all known good — every problem below is specific to SLEAP.

---

## THE OPEN BLOCKER — CUDA out of memory

**Symptom:** `torch.OutOfMemoryError` roughly 30 s into training, on both DDP ranks, inside
`torch.nn.functional.interpolate`.

**Where:** Training step only. Project creation, label import and the sanity-check pass all
succeed first — which makes early "it's running" readings misleading. (I made exactly this
mistake: GPU memory read during the sanity-check phase showed 3 GiB and looked like a fix, then
the first real training batch OOMed at 45 GiB.)

**Impact:** SLEAP cannot complete a training run at a batch size comparable to the other two
backends.

### Every configuration attempted, in order

| Backbone | Head stride | Backbone stride | Batch/GPU | Effective | Peak in use | Result |
|---|---|---|---|---|---|---|
| `unet_medium_rf` | 1 | 1 | 64 | 128 | 45.4 GiB | OOM |
| `unet_medium_rf` | 1 | 1 | 32 | 64 | 38.1 GiB | **Trained** (model collapsed — see below) |
| `convnext_tiny` | 1 | 1 | 32 | 64 | 46.1 GiB | OOM |
| `convnext_tiny` | 1 | 1 | 16 | 32 | ~47 GiB | OOM |
| `convnext_tiny` | 2 | 1 | 32 | 64 | 45.5 GiB | OOM |
| `convnext_tiny` | 2 | 2 | 32 | 64 | 45.9 GiB | OOM |
| `convnext_tiny` | 2 | 2 | 16 | 32 | ~47 GiB | OOM |

Peak figures are the values reported by the *failing* allocation, so they are lower bounds on
true peak demand.

The single trained row is the only SLEAP run that ever finished, and its model was unusable.

### Why SLEAP hits this when DLC and Lightning Pose don't

SLEAP is normally pointed at small cropped instances. Cheese3D feeds it entire camera views. The
generated config leaves frames at full size:

```yaml
scale: 1.0
max_height: null
max_width: null
```

So a 640×512 frame reaches the encoder untouched, and ConvNext-tiny's stem is configured with
`stem_patch_stride: 2` — the very first feature map is 320×256×96. At batch 32 that is about
1 GB for a single activation tensor, and dozens are retained for the backward pass.

DLC crops to 448×448 and Lightning Pose resizes, so neither pays this cost.

### What has been ruled out

- **The batch setting is applied correctly.** The generated config shows `batch_size: 16`, and
  allocation sizes track it exactly (1.88 GiB at batch 32 → 960 MiB at batch 16). Memory does
  scale with batch; the total simply stays at capacity because per-sample cost is so high.
- **Not a leak from earlier runs.** GPU memory was verified empty before each attempt. Note that
  SLEAP's DDP workers **ignore `SIGTERM`** and keep holding memory after the parent exits — they
  need `SIGKILL`. Always confirm `nvidia-smi` is clear before relaunching.
- **Output stride alone does not fix it.** Dropping the confidence-map head from stride 1 to 2
  changed peak usage by well under a gigabyte. The encoder, not the decoder, dominates.

### NOT yet tested

- Batch 8 per GPU.
- `scale: 0.5` in the preprocessing block. This is the most promising lever — it would cut
  encoder activations roughly fourfold — but it changes input resolution and therefore keypoint
  precision, which is a modelling decision rather than a settings tweak, so it was deliberately
  left alone.

### Options, roughly in order of preference

| Option | Trade-off |
|---|---|
| **`scale: 0.5`** | Halve input resolution in `PreprocessingConfig`. Biggest memory win, keeps a useful batch size. Costs spatial precision; worth checking whether SLEAP's sub-pixel peak refinement absorbs it at these keypoint sizes. |
| **batch 8** | Cheapest to try, no architectural decision needed. But effective batch 16 is a quarter of what DLC/LP used, which weakens the comparison the test exists to make. |
| **crop to ROI** | Feed SLEAP cropped mouse-face regions rather than whole frames — closer to how SLEAP is designed to be used. Most faithful to the tool, most work to wire up. |
| **gradient checkpointing** | Trades compute for activation memory without touching resolution or batch. Would need support in sleap-nn's Lightning module; not investigated. |

---

## Resolved along the way

These were hit and fixed before the memory wall. Two are latent bugs affecting every Cheese3D
user, not just this test.

### 1. Model collapse — SLEAP always trained from random initialization

**Status: fixed, but UNVERIFIED end-to-end.**

- **Symptom:** Training "succeeded" and 2D tracking reported success, but every output was NaN
  and 3D triangulation died with `ValueError: Points cannot contain NaN` inside `ConvexHull`.
- **Evidence:** 97.7% of predicted points landed on the image border, median position (1.0, 0.6)
  — the top-left corner — at confidence 0.035 against SLEAP's 0.2 detection threshold.
  Validation loss was bit-identical from epoch 19 to 99.
- **Root cause:** `_set_sleap_backbone` never set `pre_trained_weights`, which sleap-nn defaults
  to `None`. **Every SLEAP model trained through Cheese3D was randomly initialized.** On ~860
  images that converges to the trivial "predict background everywhere" solution.
- **Fix:** request ImageNet weights for each ConvNext/SwinT variant
  (`SLEAP_PRETRAINED_WEIGHTS` in `backends/sleap.py`). UNet is skipped deliberately — sleap-nn
  offers no pretrained UNet, which is why the UNet presets cannot be rescued this way.

> **Needs verification:** the fix demonstrably reaches the generated config
> (`pre_trained_weights: ConvNeXt_Tiny_Weights`), but no pretrained run has ever completed
> training because of the OOM. **Whether pretraining actually resolves the collapse is an open
> question, not a settled result.**

### 2. Backbone and head output strides could silently diverge

**Status: fixed.**

- **Root cause:** there are two independent `output_stride` settings. Selecting a backbone preset
  resets the backbone's to sleap-nn's default of 1, so the decoder upsampled from
  `max_stride: 32` to full resolution regardless of what the head asked for.
- **Fix:** `_set_sleap_backbone` now reads the head's stride and applies it to the backbone
  rather than hardcoding, so the two cannot drift apart.
- **Note:** correct, but was *not* sufficient to fix the OOM.

### 3. Lightning Pose crashed during sync — cv2 poisoning Qt's plugin path

**Status: fixed.**

- **Symptom:** `Aborted (core dumped)` during `cheese3d sync`, with "Could not load the Qt
  platform plugin xcb … even though it was found".
- **Root cause:** importing `cv2` sets `QT_QPA_PLATFORM_PLUGIN_PATH` to its own bundled Qt build.
  Matplotlib's default `qtagg` backend then initializes Qt against an incompatible plugin and
  aborts the process. It fires on figure *creation*, so merely saving a QC plot triggers it — no
  `plt.show()` needed.
- **Fix:** `clear_opencv_qt_platform_plugin_override()` in `utils.py`, called there at import and
  in `synchronize/readers.py` before matplotlib loads.

### 4. SLEAP GUI — serial crashes, then a blank video panel

**Status: fixed (committed in `65346de`).**

- **Root cause:** both PySide6 and PyQt5 are installed, and qtpy prefers PyQt5. SLEAP 1.6.1 is
  written for PySide6 — its frame loader imports PySide6 directly. Running the GUI on PyQt5
  produced five distinct type-strictness crashes, and its frame loader handed PySide6 `QImage`s
  to a PyQt5 view whose `setImage` does a strict `type(...) is QImage` check, so frames silently
  never displayed.
- **Fix:** `QT_API=pyside6` in the sleap environment (`pixi.toml`), which made all five
  per-crash patches unnecessary.

---

## Worth a second opinion

Not blocking, but each looked wrong or surprising while working through the above.

- **Channel mismatch.** The backbone config carries `in_channels: 1` while preprocessing sets
  `ensure_rgb: true` and `ensure_grayscale: false`. These appear inconsistent; never chased down.
  Could be unrelated, could matter.
- **Runtime driver.** `min_train_steps_per_epoch` — not epoch count — dominates SLEAP's runtime.
  At the measured 1.35 s/step, sleap-nn's default of 200 gives ~15 passes over 860 images per
  "epoch" and a 7.5-hour run; 50 gives ~2 hours. DLC and LP take one pass per epoch and finish in
  under half an hour.
- **Converged early.** In the one completed run, validation loss was flat from epoch 19 to 100.
  Whatever step budget is chosen, 100 epochs looks far past the point of return here.
- **Batch semantics.** `batch_size` is *per GPU* under DDP (`trainer_devices: 2`). A requested
  batch of 64 is 128 effective. Not true of DLC or Lightning Pose — an easy mistake when
  comparing backends.
- **Checkpoint semantics.** SLEAP keeps the five *best* checkpoints by validation loss
  (`best.ckpt` … `best-v4.ckpt`, plus `last.ckpt`). DLC keeps the five most recent snapshots;
  Lightning Pose keeps one per interval. Comparing "5 checkpoints" across backends is not
  comparing like with like.

---

## Code changes

Uncommitted working-tree changes:

| File | Change |
|---|---|
| `packages/cheese3d/cheese3d/backends/sleap.py` | pretrained weights, stride alignment, head stride 2 |
| `packages/cheese3d/cheese3d/utils.py` | `clear_opencv_qt_platform_plugin_override()` |
| `packages/cheese3d/cheese3d/synchronize/readers.py` | call it before matplotlib import |
| `packages/cheese3d/tests/test_sleap_backend.py` | regression test for pretrained weights |
| `test_all_backends.sh` | the end-to-end harness |

The Qt/PySide6 fixes (`pixi.toml`, `packages/cheese3d/qt_env_fix/`) are already committed in
`65346de`. All 11 SLEAP tests pass, and the new one was confirmed to fail without its fix:

```bash
pixi run -e sleap pytest packages/cheese3d/tests/test_sleap_backend.py -q
```

## Reproducing the blocker

```bash
# SLEAP only; DLC and LP are skipped via their .cheese3d_test_passed markers
ONLY=sleap ./test_all_backends.sh

# fails ~30 s into the train step
tail -f /data/disk2/home/tony/cheese3d_backend_tests/logs/c3d_sleap.log
```

The harness is resumable: a backend that completes every step is marked with
`.cheese3d_test_passed` and never rerun; a partially-completed one is rebuilt from scratch.

Deleting `cheese3d_backend_tests/c3d_sleap/` is safe. **Deleting the DLC or Lightning Pose
project directories would force a full retrain of work that already passed.**
