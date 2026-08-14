# Medium models, tier 1

Backbones of 22-50 M parameters, on one GPU. SLEAP is absent: it
has no backbone in this range.

1 epoch(s) each: enough to prove a model builds, accepts
this project's data and takes optimizer steps.

Generated: 2026-08-14 12:24

```
0, NVIDIA RTX A6000, 49140 MiB
1, NVIDIA RTX A6000, 49140 MiB
```

## DeepLabCut (1 epoch(s), batch 8, GPU 0)

| model | result | time | peak GPU | s/epoch | est. 300 epochs | est. batch @ 90 GB | detail |
|---|---|---:|---:|---:|---:|---:|---|
| `dlcrnet_stride32_ms5` | OK | 180s | 4.8 GiB | – | ? | ≥ 150 | – |

The 300-epoch estimate is startup plus per-epoch cost times 300, not
the measured run time multiplied out -- at one epoch most of the wall clock is
setup and final evaluation, so multiplying the total would be badly wrong.

Seconds per epoch comes from the gap between epoch boundaries in the log,
which needs at least two epochs to measure. With a single epoch, SLEAP and
Lightning Pose still report their own epoch duration and are timed from that,
but DeepLabCut prints no such figure and shows `–`. Run with `--epochs 3` to
time every backend properly.

The estimate assumes epochs cost the same throughout, which holds for fixed
schedules; it does not model early stopping or a changing batch size.

Peak GPU is the highest memory the run held on a single card, sampled from
`nvidia-smi` and attributed to the run's own process group.

Batch estimates come in two forms, and the prefix says which:

`~ N` is fitted. The model was run a second time at half the batch, and the
difference in peak memory gives the per-sample cost with the fixed cost
cancelled out. The estimate is then `(90 GB - fixed) / per-sample`. This
is a real maximum, though leave a margin: memory also fragments, and the last
batch that fits in a probe can still fail once a run is under way.

`>= N` is only a lower bound, produced when `--fit-batch` was not used. It
divides peak by batch, which charges the fixed cost (CUDA context, weights,
optimizer state -- none of which grow with batch) to every sample. The true
maximum is higher, and the gap is widest for the models measured at the
smallest batches.

Extrapolating across GPU generations adds error on top of that -- attention
and convolution kernels pick different algorithms and workspace sizes per
architecture, so a figure measured on one card is a guide on another, not a
guarantee.
