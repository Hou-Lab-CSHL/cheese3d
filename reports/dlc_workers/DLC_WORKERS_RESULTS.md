# DeepLabCut dataloader worker sweep

One architecture, one GPU, batch 32, 2 epoch(s).
Only the augmentation worker count changes between rows.

Read seconds per epoch, not total time: startup does not scale with
workers. `workers=-1` is DLC's own single-process default.

Cores available: 64
Generated: 2026-08-14 13:38

```
0, NVIDIA RTX A6000, 49140 MiB
1, NVIDIA RTX A6000, 49140 MiB
```

## DeepLabCut workers (2 epoch(s), batch 32, GPU 0)

| model | result | time | s/epoch | peak GPU/card | all cards | GPU busy | est. 300 epochs | max batch @ 90 GB | detail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `workers=-1` | OK | 372s | 161s | 19.8 GiB | – | 37% | 13h24m | ≥ 145 | – |
| `workers=4` | OK | 312s | 131s | 19.8 GiB | – | 43% | 10h54m | ≥ 145 | – |
| `workers=8` | OK | 299s | 127s | 19.8 GiB | – | 45% | 10h36m | ≥ 145 | – |
| `workers=16` | OK | 303s | 129s | 19.8 GiB | – | 45% | 10h44m | ≥ 145 | – |
| `workers=32` | OK | 307s | 129s | 19.8 GiB | – | 47% | 10h44m | ≥ 145 | – |
| `workers=48` | OK | 319s | 138s | 19.8 GiB | – | 43% | 11h31m | ≥ 145 | – |

**DeepLabCut workers total, 300 epochs, GPU 0, run back to back: 67h54m**

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

Peak GPU/card is the highest memory the run held on a single card, sampled
from `nvidia-smi` and attributed to the run's own process group; that per-card
figure is what a batch size has to fit inside. "All cards" sums it across
every GPU in use, for sizing a whole node, and is shown only for multi-GPU
runs.

GPU busy is the mean utilization while the run held memory. Utilization is a
device-level counter with no per-process share, so it is only meaningful
because one model runs at a time. A low figure alongside a long epoch means
the card is waiting on data loading or augmentation, not computing -- raising
the batch size will not help there.

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
