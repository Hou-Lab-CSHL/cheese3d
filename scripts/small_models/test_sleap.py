#!/usr/bin/env python
"""Small-model smoke test for the SLEAP backend.

Small means the UNet variants, which are far below ResNet-50. Measured by
building each from its sleap-nn config:

    unet            1.33 M
    unet_large_rf   1.70 M      "large" is the receptive field, not the
    unet_medium_rf  7.85 M      parameter count

SLEAP's other backbones are not close: convnext_tiny and swint_tiny both
measure ~88 M, despite the name.

The UNets train fastest of any SLEAP backbone but have no pretrained weights
in sleap-nn, so on small datasets they tend to collapse to predicting
background -- fine for a build/step check, not for a usable model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

SMALL_MODELS = ["unet", "unet_medium_rf", "unet_large_rf"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--input-scale", type=float, default=0.5,
                        help="SLEAP encodes whole frames; 1.0 exhausts a 48 GB "
                             "GPU on 640x512 views at any batch size")
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else SMALL_MODELS

    build_project(root, "small_sleap", "sleap", "sleap", testset, args.keep)

    def settings_for(model: str) -> dict:
        from cheese3d.settings import validate_training_settings
        values = {
            "epochs": args.epochs,
            # per GPU under SLEAP's DDP, unlike DLC and Lightning Pose
            "batch_size": args.batch_size,
            "learning_rate": 1e-4,
            "input_scale": args.input_scale,
            "output_stride": 2,
            "save_every_n_epochs": args.epochs,
            "validate_every_n_epochs": args.epochs,
            "backbone": model,
            "validation_fraction_percent": 10,
            "val_batch_size": min(8, args.batch_size),
            "optimizer": "Adam",
            # 0 means one pass over the data, matching DLC and LP; sleap-nn's
            # default of 200 repeats a small dataset many times per "epoch".
            "min_steps_per_epoch": 0, "steps_per_epoch": 0,
            "save_top_k": 1, "save_last": True,
            "early_stopping": False, "early_stop_patience": 10,
            "use_augmentation": True,
            "rotation_min": -15, "rotation_max": 15,
            "scale_min": 0.9, "scale_max": 1.1, "translate": 0.0,
        }
        validate_training_settings("sleap", values)
        return values

    print(f"SLEAP small models ({len(models)}), {args.epochs} epoch(s):")
    results = run_models(models, settings_for, root, "small_sleap", "sleap", args)
    return report("SLEAP", results, args)


if __name__ == "__main__":
    sys.exit(main())
