#!/usr/bin/env python
"""Small-model smoke test for the DeepLabCut backend.

Small means fewer parameters than ResNet-50, which is Cheese3D's default DLC
architecture. Measured backbone parameter counts, built from DLC's own
configs (backbone only, so ResNet-50 reads 23.51 M rather than the 25.6 M of
the full classifier):

    cspnext_s   4.38 M      resnet_50   23.51 M  <- the cutoff
    hrnet_w18   9.56 M      hrnet_w32   29.31 M
    dekr_w18    9.56 M      resnet_101  42.50 M
    cspnext_m  12.28 M      cspnext_x   47.57 M

DEKR-W18 is a different head on the same HRNet-W18 trunk, so it qualifies on
the same count.

The DLCRNet variants are not: they are multi-stage heads on a ResNet-50
trunk, so they start above the cutoff. They also fail at the evaluation stage
(``ValueError: [n, m] is not in list`` from ``prune_paf_graph``); that is
tracked separately and is not this script's job.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

SMALL_MODELS = [
    "cspnext_s",
    "hrnet_w18",
    "dekr_w18",
    "cspnext_m",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else SMALL_MODELS

    build_project(root, "small_dlc", "dlc", "dlc", testset, args.keep)

    def settings_for(model: str) -> dict:
        from cheese3d.backends.dlc import default_learning_rate
        from cheese3d.settings import validate_training_settings
        values = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            # BatchNorm backbones cannot train at ResNet's 5e-4; see
            # cheese3d.backends.dlc.default_learning_rate.
            "learning_rate": default_learning_rate(model),
            "save_every_n_epochs": args.epochs,
            "validate_every_n_epochs": args.epochs,
            "network_architecture": model,
            "train_fraction_percent": 95,
            "training_shuffle": 1,
            "max_snapshots_to_keep": 1,
            "rotation": 30, "scale_min": 0.5, "scale_max": 1.25,
            "crop_width": 448, "crop_height": 448,
            "motion_blur": False, "gaussian_noise": 12.75,
        }
        validate_training_settings("dlc", values)
        return values

    print(f"DLC small models ({len(models)}), {args.epochs} epoch(s):")
    results = run_models(models, settings_for, root, "small_dlc", "dlc", args)
    return report("DeepLabCut", results, args)


if __name__ == "__main__":
    sys.exit(main())
