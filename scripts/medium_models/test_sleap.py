#!/usr/bin/env python
"""Medium-model test for the SLEAP backend, on one GPU.

SLEAP has no genuinely medium backbone. Measured by building each from its
sleap-nn config, the gap between the UNets and everything else is enormous:

    unet_medium_rf     7.85 M      <- largest of the small suite
    swint_tiny        87.66 M      <- smallest of what is left
    convnext_tiny     87.96 M
    swint_small      108.97 M
    convnext_small   109.59 M
    swint_base       193.65 M
    convnext_base    194.47 M
    convnext_large   436.77 M      <- covered by the multi-GPU suite

So tier 1 is empty and every remaining SLEAP backbone sits in tier 2. The
"tiny" and "small" names come from the ImageNet classifiers these wrap; after
SLEAP adds its decoder, none of them are either.

Memory is the thing to watch here rather than parameter count. SLEAP encodes
whole frames, so --input-scale drives peak memory far harder than the backbone
does: at 1.0 these exhaust a 48 GB card at any batch size, which is why the
default is 0.5.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

TIERS = {
    1: [],
    2: ["swint_tiny", "convnext_tiny", "swint_small", "convnext_small",
        "swint_base", "convnext_base"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--tier", type=int, choices=(1, 2), default=2)
    parser.add_argument("--input-scale", type=float, default=0.5,
                        help="SLEAP encodes whole frames; 1.0 exhausts a 48 GB "
                             "GPU on 640x512 views at any batch size")
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else TIERS[args.tier]
    if not models:
        print(f"SLEAP has no tier {args.tier} models: its backbones jump from "
              f"7.85 M to 87.66 M with nothing in between.")
        return 0
    project = f"medium{args.tier}_sleap"

    build_project(root, project, "sleap", "sleap", testset, args.keep)

    def settings_for(model: str) -> dict:
        from cheese3d.settings import validate_training_settings
        values = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,      # per GPU under DDP
            "learning_rate": 1e-4,
            "input_scale": args.input_scale,
            "output_stride": 2,
            "save_every_n_epochs": args.epochs,
            "validate_every_n_epochs": args.epochs,
            "backbone": model,
            "validation_fraction_percent": 10,
            "val_batch_size": min(8, args.batch_size),
            "optimizer": "Adam",
            "min_steps_per_epoch": 0, "steps_per_epoch": 0,
            "save_top_k": 1, "save_last": True,
            "early_stopping": False, "early_stop_patience": 10,
            "use_augmentation": True,
            "rotation_min": -15, "rotation_max": 15,
            "scale_min": 0.9, "scale_max": 1.1, "translate": 0.0,
        }
        validate_training_settings("sleap", values)
        return values

    print(f"SLEAP medium tier {args.tier} ({len(models)} models), "
          f"{args.epochs} epoch(s), batch {args.batch_size}:")
    results = run_models(models, settings_for, root, project, "sleap", args)
    return report("SLEAP", results, args)


if __name__ == "__main__":
    sys.exit(main())
