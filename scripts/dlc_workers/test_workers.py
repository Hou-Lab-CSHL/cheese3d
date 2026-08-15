#!/usr/bin/env python
"""Find the dataloader worker count that trains DeepLabCut fastest.

One architecture, one GPU, one batch size -- only the number of augmentation
worker processes changes between runs. That isolates the input pipeline, which
is the thing being tuned: DLC3 applies albumentations transforms (rotation,
scaling, crop sampling, Gaussian noise) per sample, and with its default of
zero workers they run inside the training process while the GPU waits.

The number to compare is seconds per epoch, not total time, since startup does
not scale with workers. Two epochs suffice: per-epoch time comes from the gap
between epoch boundaries, so two give one gap -- the second epoch's duration,
measured after warmup rather than through it. GPU busy is the diagnostic beside it: a run that is
slow with a high percentage is compute-bound and more workers will not help,
while a low percentage means the card is still starved.

Two effects pull against each other, which is why this is measured rather than
reasoned about:

  * more workers augment more samples in parallel
  * DLC builds its DataLoader without persistent_workers, so every worker is
    torn down and respawned each epoch -- a cost that grows with the count and
    is paid more often the shorter the epoch

A run labelled ``workers=-1`` forces DLC's own single-process behaviour, and is
the baseline the rest are measured against.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

# -1 forces no workers (DLC's default); the rest are real counts. Three
# quarters of a 64-core machine is 48, so the ladder brackets it on both sides.
DEFAULT_COUNTS = "-1,4,8,16,32,48"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--architecture", default="hrnet_w18",
                        help="model held fixed across the sweep "
                             "(default hrnet_w18: fast, and measured at 74%% "
                             "GPU busy with no workers, so it has headroom "
                             "to show an improvement)")
    parser.add_argument("--counts", default=DEFAULT_COUNTS,
                        help=f"worker counts to compare (default {DEFAULT_COUNTS}); "
                             f"-1 means DLC's own single-process default")
    args = parser.parse_args()
    root, testset = resolve_paths(args)

    counts = [int(c) for c in args.counts.split(",") if c.strip()]
    # run_models labels each row by the "model" it is given, so the worker
    # count travels as the label and the architecture stays fixed underneath.
    labels = [f"workers={c}" for c in counts]

    build_project(root, "dlc_workers", "dlc", "dlc", testset, args.keep)

    def settings_for(label: str) -> dict:
        from cheese3d.backends.dlc import default_learning_rate
        from cheese3d.settings import validate_training_settings
        workers = int(label.split("=", 1)[1])
        values = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": default_learning_rate(args.architecture),
            "save_every_n_epochs": args.epochs,
            "validate_every_n_epochs": args.epochs,
            "network_architecture": args.architecture,
            "dataloader_workers": workers,
            "pin_memory": workers != -1,
            "train_fraction_percent": 95,
            "training_shuffle": 1,
            "max_snapshots_to_keep": 1,
            "rotation": 30, "scale_min": 0.5, "scale_max": 1.25,
            "crop_width": 448, "crop_height": 448,
            "motion_blur": False, "gaussian_noise": 12.75,
        }
        validate_training_settings("dlc", values)
        return values

    if args.epochs < 2:
        print(f"warning: --epochs {args.epochs} cannot time an epoch on its "
              f"own, and per-epoch time is the whole measurement here. Use 2+.")
    print(f"DLC dataloader workers on {args.architecture}: {counts}")
    print(f"{args.epochs} epoch(s), batch {args.batch_size}, GPU {args.gpu}")
    results = run_models(labels, settings_for, root, "dlc_workers", "dlc", args)
    return report("DeepLabCut workers", results, args)


if __name__ == "__main__":
    sys.exit(main())
