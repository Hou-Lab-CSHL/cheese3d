#!/usr/bin/env python
"""Multi-GPU test for the SLEAP backend.

Two models, for two different reasons:

    convnext_large   the open question -- it has never been confirmed to
                     train on more than one GPU, and it is the largest
                     backbone SLEAP offers
    unet_large_rf    the control -- 1.70 M parameters and known good on one
                     GPU, so if it also fails the problem is DDP or the
                     environment, not the model

SLEAP batches per GPU under DDP, so --batch-size is per card and the effective
batch is that times the GPU count: on two cards, --batch-size 16 trains on an
effective 32.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

MODELS = ["unet_large_rf", "convnext_large"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--input-scale", type=float, default=0.5,
                        help="SLEAP encodes whole frames; 1.0 exhausts a 48 GB "
                             "GPU on 640x512 views at any batch size")
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else MODELS

    build_project(root, "mgpu_sleap", "sleap", "sleap", testset, args.keep)

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

    gpus = len(args.gpu.split(","))
    print(f"SLEAP multi-GPU ({len(models)} models), {args.epochs} epoch(s), "
          f"batch {args.batch_size}/GPU x {gpus} = {args.batch_size * gpus} "
          f"effective:")
    results = run_models(models, settings_for, root, "mgpu_sleap", "sleap", args)
    return report("SLEAP", results, args)


if __name__ == "__main__":
    sys.exit(main())
