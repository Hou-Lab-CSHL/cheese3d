#!/usr/bin/env python
"""Medium-model test for the DeepLabCut backend, on one GPU.

Everything the small-model suite left behind, split by measured backbone
parameter count (built from DLC's own configs, backbone only):

    tier 1, 23-50 M          tier 2, above 50 M
    resnet_50    23.51 M     hrnet_w48   65.33 M
    hrnet_w32    29.31 M     dekr_w48    65.33 M
    dekr_w32     29.31 M
    resnet_101   42.50 M
    cspnext_x    47.57 M
    dlcrnet_stride32_ms5     (ResNet-50 trunk plus a multi-stage head)
    dlcrnet_stride16_ms5

DEKR-W32 and W48 are different heads on the HRNet trunks of the same width, so
they share those counts.

The DLCRNet pair is expected to fail, and is included so that stays visible
rather than being quietly dropped: they train, then die in evaluation with
``ValueError: [n, m] is not in list`` from prune_paf_graph. Cheese3D injects a
single-animal PAF graph when the training dataset is created, but evaluation
rebuilds the graph and cannot find those edges. They are also the one DLC
architecture pinned to a single GPU, which costs nothing here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

TIERS = {
    1: ["resnet_50", "hrnet_w32", "dekr_w32", "resnet_101", "cspnext_x",
        "dlcrnet_stride32_ms5", "dlcrnet_stride16_ms5"],
    2: ["hrnet_w48", "dekr_w48"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--tier", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else TIERS[args.tier]
    project = f"medium{args.tier}_dlc"

    build_project(root, project, "dlc", "dlc", testset, args.keep)

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

    print(f"DLC medium tier {args.tier} ({len(models)} models), "
          f"{args.epochs} epoch(s), batch {args.batch_size}:")
    results = run_models(models, settings_for, root, project, "dlc", args)
    return report("DeepLabCut", results, args)


if __name__ == "__main__":
    sys.exit(main())
