#!/usr/bin/env python
"""Timing test for DeepLabCut's HRNet, DEKR, ResNet and DLCRNet backbones.

Ten architectures on one GPU at batch 32, answering two questions: does it
run, and what would 300 epochs cost. Measured backbone parameter counts, built
from DLC's own configs:

    hrnet_w18    9.56 M     dekr_w18    9.56 M
    resnet_50   23.51 M     dekr_w32   29.31 M
    hrnet_w32   29.31 M     dekr_w48   65.33 M
    resnet_101  42.50 M     hrnet_w48  65.33 M
    dlcrnet_stride32_ms5, dlcrnet_stride16_ms5   (ResNet-50 trunk, multi-stage head)

DEKR-W18/W32/W48 are different heads on the HRNet trunks of the same width, so
they share those counts; what differs is speed, which is the point of running
both.

Three epochs by default, not one. The 300-epoch projection needs the cost of
an epoch on its own, and DeepLabCut prints no per-epoch duration -- unlike
Lightning, which SLEAP and Lightning Pose both use. The harness therefore
times the gaps between epoch boundaries in the log, which takes at least two
epochs to measure and three to have a median that ignores a slow first one.

The DLCRNet pair used to fail here, in evaluation rather than training, with
``ValueError: [n, m] is not in list`` from prune_paf_graph: Cheese3D's
keypoint groups are triangles, so the closing edge of each came out
descending, and DLC sorts spanning-tree edges before looking them up. Fixed in
_paf_graph_from_skeleton; they are kept in this list as the regression test.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

MODELS = [
    "hrnet_w18",
    "dekr_w18",
    "resnet_50",
    "hrnet_w32",
    "dekr_w32",
    "resnet_101",
    "dlcrnet_stride32_ms5",
    "dlcrnet_stride16_ms5",
    "hrnet_w48",
    "dekr_w48",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else MODELS

    build_project(root, "dlc_backbones", "dlc", "dlc", testset, args.keep)

    def settings_for(model: str) -> dict:
        from cheese3d.backends.dlc import default_learning_rate
        from cheese3d.settings import validate_training_settings
        values = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            # HRNet, DEKR and DLCRNet use BatchNorm with frozen pretrained
            # statistics and cannot train at ResNet's 5e-4: hrnet_w32 sat at a
            # flat loss until it was dropped to 1e-4. See
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

    if args.epochs < 2:
        print(f"warning: --epochs {args.epochs} cannot time an epoch on its own "
              f"(DLC prints no per-epoch duration), so the "
              f"{args.project_epochs}-epoch estimate will be '?'. Use 2 or more.")
    print(f"DLC backbones ({len(models)} models), {args.epochs} epoch(s), "
          f"batch {args.batch_size}, GPU {args.gpu}:")
    results = run_models(models, settings_for, root, "dlc_backbones", "dlc", args)
    return report("DeepLabCut", results, args)


if __name__ == "__main__":
    sys.exit(main())
