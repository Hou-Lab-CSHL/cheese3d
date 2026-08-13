# DLC3 HRNet-W32 Flat-Loss Troubleshooting — Colleague Handoff

Last reviewed: 2026-08-13

This document describes how to diagnose and potentially fix the failed DLC3 `hrnet_w32` training run. It is an experiment plan, not a record of implemented changes.

## Observed result

The first `hrnet_w32` run failed to learn:

- training loss remained near `0.0149` for more than 30 epochs;
- validation RMSE was approximately 419 px;
- validation mAP was `0.00`; and
- predictions appeared effectively random.

Using `resnet_50` with the same dataset and nominal learning rate immediately worked:

- validation RMSE: 4.18 px;
- validation mAP: 100.

This comparison strongly suggests that the labels, split, basic DLC data path, and evaluation path are usable. The likely failure is specific to HRNet initialization, BatchNorm, generated targets/configuration, gradients, or optimizer behavior.

Do not start another long HRNet run until the short diagnostics below pass.

## Primary hypotheses

### 1. HRNet pretrained weights were not loaded

This is the first item to verify.

DLC can represent initialization in several places. A backbone field such as:

```yaml
model:
  backbone:
    pretrained: false
```

does not independently prove that the model trained from scratch because initialization may also be provided through:

```text
train_settings.pretrained_weights
train_settings.weight_init
```

Checks:

- [ ] Locate the exact `pytorch_config.yaml` used by the failed run.
- [ ] Record `model.backbone`, `train_settings.pretrained_weights`, and `train_settings.weight_init`.
- [ ] Search the training log for weight download/load confirmation.
- [ ] Search for missing-key, unexpected-key, incompatible-shape, or partial-load warnings.
- [ ] Record how many checkpoint keys loaded into the backbone.
- [ ] Confirm representative HRNet tensors differ from a newly randomized model.
- [ ] Confirm only the pose head is newly initialized when that is intended.

Interpretation:

- If HRNet started randomly while ResNet loaded pretrained weights, repeat only a short HRNet test with confirmed compatible pretrained initialization.
- If the weights were downloaded but most keys were rejected, treat that as no useful initialization.
- If all expected backbone keys loaded, continue to the BatchNorm and gradient checks.

### 2. HRNet BatchNorm statistics are unstable

HRNet contains many BatchNorm layers across parallel high-resolution branches and can be more sensitive than ResNet to the batch seen by each GPU replica.

First candidate:

```yaml
freeze_bn_stats: true
freeze_bn_weights: false
```

This preserves pretrained running means and variances while leaving the learned BatchNorm scale and bias trainable.

Compare:

| Test | BN running statistics | BN affine weights |
|---|---|---|
| A | frozen | trainable |
| B | trainable | trainable |
| C | frozen | frozen |

Use A first, but only after confirming pretrained initialization. Freezing randomly initialized BatchNorm statistics could make a from-scratch model worse.

### 3. The learning rate is inappropriate for HRNet

The same learning rate is a useful controlled comparison, but different backbones do not necessarily have the same stable optimization range.

Run short, otherwise identical tests at:

- [ ] `1e-5`
- [ ] `3e-5`
- [ ] `1e-4`
- [ ] `3e-4`
- [ ] `5e-4`

Use 5–10 epochs or the tiny-set overfit test. Record batch-level loss and actual optimizer learning rate.

Interpretation:

- Lower rates learn while `5e-4` stays flat or unstable: HRNet is learning-rate sensitive.
- Every rate is perfectly flat: investigate initialization, gradients, targets, or frozen parameters.
- Very low rates move only slightly: consider warmup or separate backbone/head rates after correctness is established.

### 4. HRNet parameters may not receive or apply gradients

After one forward/backward/optimizer step, record:

- [ ] heatmap-head gradient norm;
- [ ] first HRNet-stage gradient norm;
- [ ] final HRNet-stage gradient norm;
- [ ] total gradient norm;
- [ ] count of trainable backbone parameters;
- [ ] count of trainable parameters whose gradient is `None`;
- [ ] count of nonfinite gradients; and
- [ ] before/after values for representative head and backbone tensors.

Interpretation:

| Result | Likely explanation |
|---|---|
| Head gradients exist, backbone gradients absent | Backbone frozen or disconnected |
| No useful gradients anywhere | Target/loss graph problem |
| Finite gradients exist but tensors do not change | Optimizer parameter groups or stepping problem |
| Gradients are extremely small | Sparse targets, saturation, or poor initialization |
| Gradients are nonfinite/very large | Learning rate or numerical instability |

This diagnostic is more informative than another 30-epoch run.

### 5. HRNet heatmap targets may be ineffective

Although ResNet proves the source labels are viable, the generated HRNet model may use a different output stride, target resolution, Gaussian width, or target-generator configuration.

Visualize several samples after augmentation and inspect:

- [ ] transformed input image;
- [ ] transformed keypoint coordinates;
- [ ] target heatmap dimensions;
- [ ] predicted heatmap dimensions;
- [ ] maximum target value for every keypoint;
- [ ] number/fraction of meaningful positive target pixels;
- [ ] bodypart channel order;
- [ ] keypoints excluded as invisible or outside the crop; and
- [ ] predicted heatmaps before training.

Compare the HRNet and ResNet configuration fields for:

```text
output stride
pos_dist_thresh
Gaussian sigma/target radius
heatmap mode
WeightedMSECriterion
location refinement
number and order of heatmap channels
crop dimensions
augmentation and crop sampling
```

The loss value near `0.0149` may be a background-only equilibrium. Calculate the loss produced by an all-zero prediction. If it is also approximately `0.0149`, the model has probably learned or initialized to predicting background everywhere while positive Gaussian targets contribute too little.

Only consider widening the Gaussian or increasing positive weighting after confirming the target maps visually. Do not use target changes to hide an output-shape or coordinate-conversion bug.

## Required native-config comparison

The Cheese3D GUI can send identical high-level values while DLC generates different architecture-specific native settings.

Diff the complete successful ResNet and failed HRNet `pytorch_config.yaml` files. At minimum compare:

```text
model.backbone
model.heads
train_settings.pretrained_weights
train_settings.weight_init
train_settings.batch_size
runner.optimizer
runner.scheduler
freeze_bn_stats
freeze_bn_weights
data.train
data.train.crop_sampling
target_generator
heatmap_config
```

Save the diff with the experiment results. Do not assume GUI equality means native-config equality.

## Most informative experiment: overfit 16–32 images

Before full-dataset training:

1. Select 16–32 clean, representative labeled images.
2. Use one GPU.
3. Disable or substantially reduce augmentation.
4. Load confirmed compatible HRNet pretrained weights.
5. Freeze BatchNorm running statistics but keep affine weights trainable.
6. Start with learning rate `1e-4`.
7. Train and evaluate on the same tiny set.
8. Inspect heatmaps and overlays, not just aggregate loss.

A correctly connected pose model should be able to memorize this set and drive training error close to zero.

Interpretation:

- Cannot memorize 16 images: initialization, target generation, gradients, optimizer, frozen parameters, BatchNorm, or an HRNet/DLC integration defect.
- Memorizes without augmentation but fails with it: augmentation/crop problem.
- Memorizes on one GPU but not two: multi-GPU, device placement, gradient reduction, or per-rank BatchNorm problem.
- Memorizes the tiny set and learns the full training set but validation remains random: split/domain/generalization problem.

## Single-GPU before multi-GPU

Establish correct HRNet learning on one GPU before restoring two-GPU training.

If single-GPU works and multi-GPU stays flat, investigate:

- [ ] BatchNorm statistics per replica;
- [ ] synchronized BatchNorm support and behavior;
- [ ] correct model and input device placement;
- [ ] optimizer creation relative to model wrapping;
- [ ] gradient reduction across ranks/devices;
- [ ] effective learning-rate scaling;
- [ ] duplicate or uneven sampling; and
- [ ] whether every rank starts with identical pretrained weights.

Two GPUs increase throughput but do not help determine whether the architecture is correctly learning.

## Recommended initial HRNet candidate

After initialization is verified:

```text
Architecture: hrnet_w32
Initialization: confirmed compatible pretrained HRNet weights
GPU count: 1 for diagnosis
Batch size: 8–16
Learning rate: 1e-4
BN statistics: frozen
BN affine weights: trainable
Augmentation: minimal for tiny-set overfit test
Run length: 5–10 epochs initially
```

If `1e-4` remains flat, try `3e-5` and `1e-5`. Do not spend another long run at `5e-4` until the initialization, targets, gradients, and parameter updates are proven correct.

## Order of operations

Perform the investigation in this order:

1. [ ] Locate and archive the exact failed HRNet and successful ResNet configs/logs.
2. [ ] Diff their complete native `pytorch_config.yaml` files.
3. [ ] Prove whether HRNet pretrained weights loaded successfully.
4. [ ] Inspect trainable parameter counts and optimizer parameter groups.
5. [ ] Run one forward/backward/step and record gradients and tensor changes.
6. [ ] Visualize HRNet target heatmaps for multiple augmented samples.
7. [ ] Calculate the all-zero-prediction baseline loss.
8. [ ] Overfit 16–32 images on one GPU with minimal augmentation.
9. [ ] Repeat with confirmed pretraining and frozen BN statistics.
10. [ ] Run the short learning-rate sweep.
11. [ ] Reintroduce augmentations one component at a time.
12. [ ] Test the full dataset on one GPU.
13. [ ] Restore two-GPU training and compare behavior.

## Evidence to save for each run

- [ ] Git commit and dirty-tree status.
- [ ] Pixi environment and DLC/PyTorch/CUDA versions.
- [ ] Full project and `pytorch_config.yaml` files.
- [ ] Shuffle, train fraction, and exact train/validation image lists.
- [ ] Architecture and initialization source.
- [ ] Missing/unexpected checkpoint keys.
- [ ] GPU IDs and number of devices.
- [ ] Physical and effective batch size.
- [ ] Actual learning rate over time.
- [ ] BatchNorm freeze settings.
- [ ] Enabled augmentations and crop size.
- [ ] Gradient/parameter-update diagnostic.
- [ ] Batch-level and epoch-level loss.
- [ ] Validation RMSE and mAP.
- [ ] Example target heatmaps, prediction heatmaps, and overlays.

## Decision tree

```text
Did compatible pretrained HRNet weights load?
├── No  -> fix initialization, then repeat tiny-set test
└── Yes
    └── Do head and backbone receive finite gradients and update?
        ├── No  -> inspect freezing, optimizer groups, graph, and loss targets
        └── Yes
            └── Can one GPU overfit 16–32 images?
                ├── No  -> test BN freeze, target heatmaps, LR sweep
                └── Yes
                    └── Does full data learn with minimal augmentation?
                        ├── No  -> inspect dataset diversity/sampling/split
                        └── Yes
                            └── Does enabling augmentation break it?
                                ├── Yes -> isolate crop/augmentation component
                                └── No
                                    └── Does two-GPU training break it?
                                        ├── Yes -> multi-GPU/BN/reduction problem
                                        └── No  -> proceed with full training
```

## Likelihood ranking

Current best estimate, before the diagnostics:

1. HRNet did not receive useful pretrained initialization.
2. HRNet BatchNorm statistics were unsuitable for the per-device batch/workflow.
3. `5e-4` is outside the stable HRNet learning-rate range.
4. HRNet's generated target/output configuration permits a background-only solution.
5. Backbone parameters are frozen, disconnected, or absent from the optimizer.
6. Multi-GPU wrapping or reduction is architecture-sensitive.

The fastest discriminator is a confirmed-pretrained, frozen-BN, single-GPU tiny-set overfit test with gradient and target inspection.

## Official reference

- [DeepLabCut training guide](https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md)
- [DeepLabCut BatchNorm backbone behavior](https://deeplabcut.github.io/DeepLabCut/dev/3.0/reference/deeplabcut/pose_estimation_pytorch/models/backbones/base/)
- [DeepLabCut official repository](https://github.com/DeepLabCut/DeepLabCut)

