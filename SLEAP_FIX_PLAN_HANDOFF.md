# SLEAP Memory and Model-Quality Fix Plan — Colleague Handoff

Last reviewed: 2026-08-13

This document follows [SLEAP_HANDOFF.md](SLEAP_HANDOFF.md). It records a second investigation of the blocker using the Cheese3D adapter, the locally installed SLEAP-NN 0.1.0 source, official SLEAP/SLEAP-NN documentation, upstream releases, and developer recommendations.

No implementation changes or GPU tests were performed during this investigation.

## Executive summary

SLEAP's current CUDA out-of-memory failure is most likely a configuration/workflow mismatch rather than a fundamental inability to process this dataset:

- Cheese3D passes complete `640×512` camera views to SLEAP.
- Preprocessing currently uses `scale: 1.0`, so the encoder receives the full image.
- Pretrained ConvNeXt and SwinT backbones require three-channel RGB.
- Large encoder and decoder activations must remain resident for backpropagation.
- SLEAP's configured batch size is **per GPU**. Batch 16 with two GPUs is an effective batch of 32.
- DDP duplicates the model on each GPU. It does not combine two 47.4 GiB devices into one 94.8 GiB memory pool.

The first fix to test is SLEAP's supported input preprocessing scale, especially `scale: 0.5`, combined with a realistic batch of 4 or 8 per GPU. Official documentation presents half-resolution input as a normal performance option. If downscaling loses unacceptable keypoint precision, the next architectural solution should be fixed per-camera ROIs or SLEAP's crop-based top-down workflow.

## Evidence from the installed implementation

The active environment contains:

- SLEAP 1.6.1
- SLEAP-NN 0.1.0
- sleap-io 0.6.4
- PyTorch 2.7.1 with CUDA 12.8 wheels

The installed SLEAP-NN preprocessing configuration supports:

```yaml
data_config:
  preprocessing:
    ensure_rgb: true
    ensure_grayscale: false
    scale: 1.0
    max_height: null
    max_width: null
```

`scale` resizes both images and coordinates in the training dataset. SLEAP's inference pipeline stores the effective/input scale and reverses the coordinate transform when returning predictions to original-image space.

### Why `scale: 0.5` is the primary lever

For the current input:

| Setting | Model input | Pixels per image | Relative area |
|---|---:|---:|---:|
| `scale: 1.0` | `640×512` | 327,680 | 1.00 |
| `scale: 0.75` | `480×384` | 184,320 | 0.56 |
| `scale: 0.5` | `320×256` | 81,920 | 0.25 |

Half scaling should reduce the dominant spatial activation memory by roughly fourfold. Exact peak memory will not fall by precisely four because weights, optimizer states, callbacks, confidence maps, and CUDA workspace allocations remain.

SLEAP-NN's official data guide explicitly lists `scale: 0.5` as the common downscaling pattern and describes it as approximately four times faster.

### Batch size semantics

SLEAP-NN's batch size is per DDP rank/GPU:

```text
effective batch = configured batch per GPU × number of GPUs
```

Therefore:

| Batch/GPU | GPUs | Effective batch |
|---:|---:|---:|
| 4 | 2 | 8 |
| 8 | 2 | 16 |
| 16 | 2 | 32 |
| 32 | 2 | 64 |

Official SLEAP configuration examples use a batch size of 4. Matching DLC or Lightning Pose's numeric batch value is not a fair requirement because their input processing, architecture, confidence maps, and optimization loops differ.

### Mixed precision and gradient accumulation

SLEAP-NN 0.1.0 creates its Lightning trainer without passing `precision` or `accumulate_grad_batches`. Those fields are also absent from the published 0.1.x `TrainerConfig`.

Consequently, adding an unknown YAML key is not sufficient. Supporting these features would require either:

1. upgrading to an upstream version that officially exposes them; or
2. implementing a narrow Cheese3D/SLEAP-NN trainer wrapper or patch.

Mixed precision is worth considering after input scaling. Gradient accumulation can emulate a larger optimization batch but does not reduce the memory needed for one sample; it only permits a smaller physical batch.

### Channel mismatch is corrected upstream

The handoff noted that a generated backbone may initially show `in_channels: 1` while preprocessing says `ensure_rgb: true`.

SLEAP-NN's `ModelTrainer._verify_model_input_channels()` corrects this during setup. When ConvNeXt or SwinT pretrained weights are selected, it forces:

```text
backbone in_channels = 3
ensure_rgb = true
ensure_grayscale = false
```

This mismatch is confusing in the initial YAML but is not the OOM cause. The effective config saved in the model output should still be checked to prove that the runtime correction occurred.

### Output stride is secondary

Aligning the backbone decoder and confidence-map head output strides is correct and should remain. The measured runs show that changing output stride alone saved little compared with the full-frame encoder cost. Do not treat a larger stride as the primary fix.

### Epochs are not comparable between backends

SLEAP-NN calculates the number of training steps using:

```text
steps per epoch = max(natural batches in dataset, min_train_steps_per_epoch)
```

The default minimum is 200. With approximately 860 training images, this can repeat the dataset many times inside one nominal SLEAP epoch. DLC and Lightning Pose's epoch semantics are different.

Backend comparisons should use:

- total samples processed;
- optimizer steps;
- wall-clock training time; and
- validation quality.

They should not rely only on the displayed epoch count.

## Recommended implementation plan

### Phase 1 — Expose and explain the actual memory controls

- [ ] Add SLEAP preprocessing input scale to the Cheese3D GUI and config.
- [ ] Offer at least `1.0`, `0.75`, and `0.5`.
- [ ] Label training batch size explicitly as **batch per GPU**.
- [ ] Display the computed effective global batch.
- [ ] Display original and processed image dimensions before training.
- [ ] Keep the backbone/head output-stride controls synchronized.
- [ ] Expose natural/exact steps per epoch and minimum steps per epoch clearly.
- [ ] Print one concise launch summary to the terminal.

Suggested terminal summary:

```text
SLEAP input: 640×512 RGB -> 320×256 RGB (scale 0.5)
SLEAP batch: 8/GPU × 2 GPUs = 16 effective
SLEAP epoch: 54 global batches; minimum override disabled
SLEAP model: pretrained ConvNeXt-tiny, confidence-map stride 2
```

### Phase 2 — Establish a controlled memory/quality baseline

Do not begin with a full 100-epoch run. Once GPU validation is authorized, run 5–10 epochs per candidate:

| Test | Scale | Batch/GPU | GPUs | Purpose |
|---|---:|---:|---:|---|
| A | 1.0 | 4 | 1 | Full-resolution minimum-memory baseline |
| B | 1.0 | 8 | 2 | Determine whether scaling is required |
| C | 0.75 | 8 | 2 | Intermediate memory/accuracy point |
| D | 0.5 | 8 | 2 | Most promising practical setting |
| E | 0.5 | 16 | 2 | Determine remaining throughput margin |

Record for every run:

- [ ] exact environment and package versions;
- [ ] processed tensor shape;
- [ ] physical and effective batch sizes;
- [ ] peak allocated and reserved VRAM on every rank;
- [ ] seconds per optimizer step;
- [ ] training and validation loss;
- [ ] validation pixel RMSE, PCK, mOKS or other available pose metrics;
- [ ] prediction confidence distribution;
- [ ] missing/NaN keypoint percentage;
- [ ] percentage of predictions on the image border;
- [ ] triangulation success and reprojection error; and
- [ ] visual overlay inspection.

### Phase 3 — Correct the default training duration

- [ ] Calculate the natural number of batches from dataset size and effective batch.
- [ ] Stop silently forcing 200 steps per epoch for a dataset this size.
- [ ] Default to one natural dataset pass per epoch unless the user explicitly requests a minimum.
- [ ] Keep exact steps as an advanced override.
- [ ] Enable early stopping by default.
- [ ] Compare backends by samples/steps and wall time, not nominal epochs.

### Phase 4 — Verify that pretrained models solve the collapse

The pretrained-weight patch reaches the generated config but has not completed a training run. Treat it as unverified.

Acceptance criteria:

- [ ] Training and validation loss decrease meaningfully.
- [ ] Validation loss does not become bit-identical for most of the run.
- [ ] Predicted confidence rises above the configured inference threshold.
- [ ] Predictions do not concentrate on image borders or the top-left corner.
- [ ] Missing predictions remain below an agreed threshold.
- [ ] Labeled-video overlays align with the animal.
- [ ] Triangulation completes without all-NaN frames.

If pretrained ConvNeXt still collapses, investigate:

- [ ] whether ImageNet weights were actually loaded, not merely named in YAML;
- [ ] pretrained image normalization requirements;
- [ ] confidence-map sigma after input downscaling;
- [ ] learning rate and warmup;
- [ ] missing/visibility label handling;
- [ ] positive Gaussian target versus background imbalance; and
- [ ] whether augmentation moves keypoints outside the image excessively.

### Phase 5 — Add crop-based training if downscaling loses precision

There are two possible designs.

#### Option A: fixed Cheese3D per-camera ROIs

Apply existing camera crops consistently during:

- label conversion;
- training;
- inference; and
- coordinate restoration before writing the HDF5 result.

This is probably the simplest solution for one animal and fixed camera geometry.

#### Option B: native SLEAP top-down pipeline

Train:

1. a whole-frame centroid model; and
2. a centered-instance model on small crops.

This matches SLEAP's recommended approach when an animal occupies a small part of the full image, but it requires two models, paired checkpoint selection, two-stage inference, and more GUI/config work.

Decision rule:

- Prefer fixed ROIs if camera geometry supplies a stable region containing the animal.
- Prefer native top-down if animal position varies substantially and a fixed ROI cannot remain small.

### Phase 6 — Add trainer-level memory features only if needed

If reasonable scaling/crops and batch 4–8 still fail:

- [ ] expose Lightning mixed precision (`16-mixed` or `bf16-mixed`);
- [ ] expose gradient accumulation;
- [ ] test numerical stability and prediction quality;
- [ ] investigate activation checkpointing for ConvNeXt/Swin blocks; and
- [ ] add a preflight memory estimator.

Priority within this phase:

1. mixed precision;
2. gradient accumulation;
3. activation checkpointing.

Activation checkpointing is the most invasive and should not be the first intervention.

### Phase 7 — Test a newer SLEAP stack separately

Official upstream guidance now points to a newer combination:

- SLEAP 1.6.3
- SLEAP-NN 0.2.0
- sleap-io 0.7.0

SLEAP-NN 0.2.0 and sleap-io 0.7.0 contain API-breaking changes. Do not replace the known environment in place. Create a separate Pixi environment such as `sleap-next` and validate:

- [ ] Cheese3D label conversion;
- [ ] SLP read/write compatibility;
- [ ] generated config schema;
- [ ] SLEAP labeling GUI startup;
- [ ] single-GPU training;
- [ ] multi-GPU DDP training;
- [ ] early-stop and DDP worker shutdown;
- [ ] progress output;
- [ ] checkpoint discovery and validation metrics;
- [ ] single- and multi-camera inference;
- [ ] original-coordinate restoration after scaling/cropping; and
- [ ] HDF5 conversion and triangulation.

Upstream releases after 0.1.0 include fixes affecting multi-GPU GUI launches, DDP shutdown/control, configuration generation, caching, inference, and progress reporting. An upgrade may remove several Cheese3D workarounds, but it should remain independent from the immediate memory experiment.

## Changes that should not be attempted first

- Do not add more GPUs expecting their VRAM to pool under DDP.
- Do not raise confidence-map stride repeatedly while leaving full-frame input unchanged.
- Do not compare the same numeric batch across different pose frameworks as if the memory workload were equivalent.
- Do not start another 100-epoch run before a short run proves that loss and predictions improve.
- Do not implement activation checkpointing before testing supported input scaling.
- Do not upgrade SLEAP, SLEAP-NN, and sleap-io directly inside the existing validated environment.
- Do not declare the pretrained-weight fix successful until inference quality and triangulation are measured.

## Recommended first working configuration

The best initial candidate is:

```yaml
data_config:
  preprocessing:
    ensure_rgb: true
    ensure_grayscale: false
    scale: 0.5

model_config:
  backbone_config:
    convnext:
      pre_trained_weights: ConvNeXt_Tiny_Weights
      output_stride: 2
  head_configs:
    single_instance:
      confmaps:
        output_stride: 2

trainer_config:
  train_data_loader:
    batch_size: 8       # per GPU
  val_data_loader:
    batch_size: 8
  trainer_devices: 2
  trainer_strategy: ddp
  min_train_steps_per_epoch: 0  # use natural dataset length in Cheese3D logic
  max_epochs: 10                # initial validation only
```

The exact accepted representation of a disabled minimum must be verified against SLEAP-NN's validator. If zero is rejected or still overrides incorrectly, Cheese3D should calculate and write the natural step count explicitly.

## Official references

- [SLEAP-NN data configuration and preprocessing](https://nn.sleap.ai/v0.1.3/configuration/data/)
- [SLEAP-NN preprocessing implementation](https://nn.sleap.ai/latest/api/inference/layers/)
- [SLEAP-NN coordinate preprocessing metadata](https://nn.sleap.ai/dev/api/inference/preprocess_info/)
- [SLEAP-NN trainer configuration](https://nn.sleap.ai/v0.1.0/configuration/trainer/)
- [SLEAP-NN 0.1.3 trainer API](https://nn.sleap.ai/v0.1.3/api/config/trainer_config/)
- [SLEAP-NN releases](https://github.com/talmolab/sleap-nn/releases/)
- [Official SLEAP repository](https://github.com/talmolab/sleap)
- [SLEAP developer guidance on large images and top-down crops](https://github.com/talmolab/sleap/issues/2230)

## Handoff order of operations

1. Expose preprocessing scale and processed dimensions.
2. Clarify per-GPU/effective batch semantics.
3. Correct the 200-step epoch behavior.
4. Run pretrained ConvNeXt-tiny at scale 0.5 and batch 8/GPU for 5–10 epochs.
5. Evaluate pose quality and downstream triangulation, not merely training completion.
6. Add fixed ROIs if half-resolution accuracy is insufficient.
7. Consider mixed precision and gradient accumulation only if memory remains limiting.
8. Evaluate SLEAP 1.6.3/SLEAP-NN 0.2.0 in an isolated `sleap-next` environment.

