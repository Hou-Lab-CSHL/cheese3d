# Which models work

Every architecture Cheese3D offers was trained for one epoch on the 3,052-image
demo project (640×512, 28 keypoints, 2 × RTX A6000). **42 of 44 work.**

Last verified: 2026-08-13 · SLEAP 1.6.1 / sleap-nn 0.1.0 · DLC 3.0.1 · Lightning Pose 2.2

| Backend | Working | Blocked |
|---|---|---|
| DeepLabCut | 13 / 13 | — |
| SLEAP | 10 / 10 | — |
| Lightning Pose | 19 / 21 | 2 (DINOv3 licence) |

## DeepLabCut — all 13 work

`resnet_50` `resnet_101` · `hrnet_w18` `hrnet_w32` `hrnet_w48` ·
`dekr_w18` `dekr_w32` `dekr_w48` · `cspnext_s` `cspnext_m` `cspnext_x` ·
`dlcrnet_stride16_ms5` `dlcrnet_stride32_ms5`

**Recommended: `resnet_50`.** DLC3's default and the one its default learning
rate is tuned for; reached RMSE 4.18 px / mAP 100 in 26 min. `hrnet_w32` was
tried first and its loss stayed flat at this learning rate, so HRNet needs its
own LR tuning before use.

Speed at one epoch: cspnext ~85 s, resnet ~100 s, hrnet ~220 s, dekr ~260 s.

**DLCRNet is forced onto one GPU.** Its multi-scale branches are not replicated
correctly by DLC's DataParallel wrapper, so multi-GPU training dies in ~19 s
with "Expected all tensors to be on the same device". Cheese3D detects this and
falls back automatically — no action needed, just expect it to be slower.

**Top-down architectures were removed** from the menu (`top_down_*`,
`rtmpose_*`, `ctd_*`, `animaltokenpose_base`). They crop to a detected bounding
box instead of sampling from the full frame, and the CTD family additionally
needs a prior pose as a conditioning input. Cheese3D has no detector stage, so
none apply. Restoring them means adding a detector, not flipping a flag.

## SLEAP — all 10 work

`unet` `unet_medium_rf` `unet_large_rf` · `convnext_tiny` `convnext_small`
`convnext_base` `convnext_large` · `swint_tiny` `swint_small` `swint_base`

**Recommended: `convnext_tiny`.** UNet variants train but have no pretrained
weights available in sleap-nn, and from random init on a few thousand images
they collapse to predicting background everywhere — inference then puts ~98% of
points on the image border below the 0.2 detection threshold, so every output
is NaN. Prefer a ConvNeXt or SwinT backbone, which load ImageNet weights.

**Always set `input_scale: 0.5` (or lower).** SLEAP runs its encoder over whole
640×512 camera views, unlike DLC (448×448 crops) and LP (resize). At scale 1.0
it exhausts a 47 GiB GPU at *every* batch size. At 0.5 it uses 9.3 GiB at batch
8. Labels are rescaled with the images and predictions come back in
original-frame coordinates.

`convnext_large` needs batch ≤ 8/GPU (30.8 GiB); it OOMs at 16.

**Batch size is per GPU** under SLEAP's DDP — 40 with two GPUs means an
effective batch of 80. DLC and LP take the number as given.

Speed at one epoch, scale 0.5: unet ~75 s, swint 150–270 s, convnext 210–350 s.

## Lightning Pose — 19 of 21 work

Working: `resnet18` `resnet34` `resnet50` `resnet101` `resnet152` ·
`resnet50_animal_apose` `resnet50_animal_ap10k` ·
`efficientnet_b0` `efficientnet_b1` `efficientnet_b2` ·
`vits_dino` `vits_dinov2` `vitb_dino` `vitb_dinov2` `vitb_imagenet`

**Recommended: `resnet50_animal_ap10k`.** Pretrained on the AP-10K animal-pose
dataset. The fastest backend overall: ResNets and EfficientNets finish an epoch
in 49–117 s.

**ViT/DINO backbones are forced onto one GPU.** Training works across GPUs, but
the post-training prediction pass then hangs forever — one GPU pinned at 100%
holding 1.4 GiB with zero batches, an NCCL spin-wait on a peer that never
rejoins. On one GPU the same model predicts 93 batches in 25 s. Cheese3D falls
back automatically. ViTs also force square 512×512 input.

### Blocked

| Model | Reason | Fix |
|---|---|---|
| `vits_dinov3`, `vitb_dinov3` | Meta gates DINOv3 weights | Accept the licence on Hugging Face and authenticate (`huggingface-cli login`) |

`vitb_sam` was removed from the menu.

## Re-checking this

```bash
./sweep_models.sh                # every architecture, one epoch each (~2 h)
ONLY=sleap ./sweep_models.sh     # one backend
```

Results land in `../cheese3d_model_sweep/MODEL_SWEEP_RESULTS.md`. Paths derive
from the script's location; set `TESTSET=/path/to/data` for a different layout.

Note that `sweep_models.sh` measures *compatibility*, not accuracy — one epoch
only proves a model builds, accepts the data, and takes optimizer steps.
