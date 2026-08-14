#!/usr/bin/env python
"""Small-model smoke test for the Lightning Pose backend.

Small means fewer parameters than ResNet-50 (25.6 M):

    efficientnet_b0  5.3 M     resnet18  11.7 M
    efficientnet_b1  7.8 M     resnet34  21.8 M
    efficientnet_b2  9.1 M

The ViT-S backbones sit just under that parameter mark but are not
lightweight to run -- attention over the full frame makes them several times
slower per epoch than anything here -- so they belong in the full sweep, not
in this quick check.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

SMALL_MODELS = [
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "resnet18",
    "resnet34",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--dinov3-weights", default="",
                        help="directory with locally downloaded DINOv3 models; "
                             "without it the gated Hugging Face repo is used")
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else SMALL_MODELS
    dinov3 = args.dinov3_weights or str(testset)

    build_project(root, "small_lp", "lightning_pose", "lp", testset, args.keep)

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
            # ViTs keep the backbone frozen until this epoch; leaving it at the
            # usual 20 means a short run trains only the head and predicts
            # nothing above the detection threshold.
            "unfreezing_epoch": 2 if model.startswith("vit") else 20,
            "early_stopping": False, "early_stop_patience": 3,
            "dinov3_weights_dir": dinov3,
        }
        validate_training_settings("lightning_pose", values)
        return values

    print(f"Lightning Pose small models ({len(models)}), {args.epochs} epoch(s):")
    results = run_models(models, settings_for, root, "small_lp", "lp", args)
    return report("Lightning Pose", results, args)


if __name__ == "__main__":
    sys.exit(main())
