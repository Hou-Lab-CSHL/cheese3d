#!/usr/bin/env python
"""Multi-GPU test for the Lightning Pose DINO backbones.

These train under DDP, but Lightning Pose's own post-training prediction pass
deadlocks for ViTs on more than one GPU -- rank 0 blocks forever in an NCCL
collective waiting for a peer that has already gone. Cheese3D passes
skip_evaluation for exactly that case, so what this checks is that the
workaround holds across the whole DINO family, rather than only on the one
configuration it was written against.

An OK row therefore means training completed with evaluation skipped by
design. Predictions come from `cheese3d track` afterwards, in a fresh
single-GPU process.

vitl_dinov3 needs locally downloaded weights: the Hugging Face repo is gated.
Pass --dinov3-weights, or let it default to the test-set directory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

MODELS = [
    "vits_dinov2",
    "vits_dinov3",
    "vitb_dinov2",
    "vitb_dinov3",
    "vitl_dinov3",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--dinov3-weights", default="",
                        help="directory with locally downloaded DINOv3 models; "
                             "without it the gated Hugging Face repo is used")
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else MODELS
    dinov3 = args.dinov3_weights or str(testset)

    build_project(root, "mgpu_lp", "lightning_pose", "lp", testset, args.keep)

    def settings_for(model: str) -> dict:
        from cheese3d.settings import validate_training_settings
        values = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            # LP asserts ckpt_every_n_epochs % check_val_every_n_epoch == 0.
            "save_every_n_epochs": args.epochs,
            "validate_every_n_epochs": args.epochs,
            "backbone": model,
            "imgaug": "default",
            "horizontal_flip": False,
            "train_prob": 0.95, "val_prob": 0.05,
            # ViTs keep the backbone frozen until this epoch; the usual 20
            # would leave a short run training only the head.
            "unfreezing_epoch": min(2, args.epochs),
            "early_stopping": False, "early_stop_patience": 3,
            "dinov3_weights_dir": dinov3,
        }
        validate_training_settings("lightning_pose", values)
        return values

    print(f"Lightning Pose multi-GPU ({len(models)} models), "
          f"{args.epochs} epoch(s), batch {args.batch_size}:")
    results = run_models(models, settings_for, root, "mgpu_lp", "lp", args)
    return report("Lightning Pose", results, args)


if __name__ == "__main__":
    sys.exit(main())
