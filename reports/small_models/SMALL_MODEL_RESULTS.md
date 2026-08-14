# Small-model smoke test

Architectures below ResNet-50's 25.6 M parameters, 1 epoch(s) each.
This checks that a model builds, accepts the data and takes optimizer
steps -- not that it learns anything.

Generated: 2026-08-13 22:29

## DeepLabCut (1 epoch(s), GPU 0)

| model | result | time | detail |
|---|---|---:|---|
| `cspnext_s` | OK | 82s | – |
| `cspnext_m` | OK | 90s | – |

## Lightning Pose (1 epoch(s), GPU 0)

| model | result | time | detail |
|---|---|---:|---|
| `efficientnet_b0` | OK | 69s | – |
| `efficientnet_b1` | OK | 82s | – |
| `efficientnet_b2` | OK | 85s | – |
| `resnet18` | OK | 54s | – |
| `resnet34` | OK | 81s | – |

## SLEAP (1 epoch(s), batch 8, GPU 0)

| model | result | time | peak GPU | est. batch @ 90 GB | detail |
|---|---|---:|---:|---:|---|
| `unet` | OK | 71s | 1.4 GiB | ≥ 512 | – |

Peak GPU is the highest memory the run held on a single card, sampled from
`nvidia-smi` and attributed to the run's own process group.

The batch estimate divides that peak by the batch size actually used, then
divides 90 GB by the result. It is a **floor, not a ceiling**: peak
memory is a fixed cost (CUDA context, weights, optimizer state, which do not
grow with batch) plus a per-sample cost, and charging the fixed part to every
sample overstates what one sample needs. The true ceiling is higher, and the
gap is widest for the models with the smallest batches.

To turn the floor into a real number, run the same model at two batch sizes:
the difference in peak divided by the difference in batch is the per-sample
cost with the fixed part cancelled out.

Extrapolating across GPU generations adds error on top of that -- attention
and convolution kernels pick different algorithms and workspace sizes per
architecture, so a figure measured on one card is a guide on another, not a
guarantee.
