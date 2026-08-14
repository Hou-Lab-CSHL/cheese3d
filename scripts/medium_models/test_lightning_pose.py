#!/usr/bin/env python
"""Medium-model test for the Lightning Pose backend, on one GPU.

Everything the small-model suite left behind, minus the DINO ViTs, which are
covered by the multi-GPU suite instead. Split by parameter count:

    tier 1, 22-45 M                     tier 2, above 45 M
    vits_dino                ~22 M      resnet152          60.2 M
    resnet50                 25.6 M     vitb_dino          ~86 M
    resnet50_animal_ap10k    25.6 M     vitb_imagenet      ~86 M
    resnet50_animal_apose    25.6 M
    resnet50_human_hand      25.6 M
    resnet50_human_jhmdb     25.6 M
    resnet50_human_res_rle   25.6 M
    resnet50_human_top_res   25.6 M
    resnet101                44.5 M

The six pretrained resnet50 variants are the same architecture with different
initial weights, so they cost the same to run; what differs is how well they
transfer, which only a real training run shows.

vits_dino belongs here rather than in the small suite despite its size: at
~22 M it is under ResNet-50's parameter count, but attention over the full
frame makes it several times slower per epoch than any small model.

vitb_sam is deliberately absent -- it was dropped from Cheese3D earlier.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (add_common_arguments, build_project, report,  # noqa: E402
                    resolve_paths, run_models)

TIERS = {
    1: ["resnet50", "resnet50_animal_ap10k", "resnet50_animal_apose",
        "resnet50_human_hand", "resnet50_human_jhmdb",
        "resnet50_human_res_rle", "resnet50_human_top_res",
        "resnet101", "vits_dino"],
    2: ["resnet152", "vitb_dino", "vitb_imagenet"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--tier", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dinov3-weights", default="",
                        help="directory with locally downloaded DINOv3 models; "
                             "unused by this tier unless --models names one")
    args = parser.parse_args()
    root, testset = resolve_paths(args)
    models = args.models.split(",") if args.models else TIERS[args.tier]
    project = f"medium{args.tier}_lp"
    dinov3 = args.dinov3_weights or str(testset)

    build_project(root, project, "lightning_pose", "lp", testset, args.keep)

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
            "unfreezing_epoch": min(2, args.epochs) if model.startswith("vit") else 20,
            "early_stopping": False, "early_stop_patience": 3,
            "dinov3_weights_dir": dinov3,
        }
        validate_training_settings("lightning_pose", values)
        return values

    print(f"Lightning Pose medium tier {args.tier} ({len(models)} models), "
          f"{args.epochs} epoch(s), batch {args.batch_size}:")
    results = run_models(models, settings_for, root, project, "lp", args)
    return report("Lightning Pose", results, args)


if __name__ == "__main__":
    sys.exit(main())
