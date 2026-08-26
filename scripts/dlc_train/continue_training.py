#!/usr/bin/env python
"""Continue training the DLC models, selecting on test RMSE.

Copies each finished project to a new directory and trains on from its best
snapshot, so the original weights, logs and evaluation results stay exactly as
they are. Nothing under --source is written to.

Two things differ from the first run:

  * the best snapshot is chosen by test.rmse rather than test.mAP. mAP
    saturated around 99.4 on this data by epoch 90 and its remaining movement
    was noise, so its argmax picked the luckiest evaluation; rmse was still
    falling at epoch 240 and keeps resolving.
  * training resumes from the best snapshot rather than starting cold.

What resuming does and does not restore matters here. DLC loads the model
weights and nothing else -- ``model.load_state_dict(snapshot["model"])`` -- so
the optimizer's moment estimates, the learning-rate schedule position and the
epoch counter all start fresh. This is a warm start, not a seamless
continuation: expect a transient loss bump while Adam re-estimates, and the
requested epochs are trained in full rather than counted from where the
previous run stopped.

The first run also pruned every snapshot except the best, so the best is the
only available resume point; the final epoch's weights no longer exist.

    python continue_training.py --models hrnet_w48 --epochs 120
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/data/disk2/home/tony/dlc_projects")
DEST = Path("/data/disk2/home/tony/dlc_projects_continued")

MODELS = ["hrnet_w48", "hrnet_w18", "resnet_50", "dekr_w32",
          "dlcrnet_stride32_ms5", "dekr_w48", "dekr_w18", "resnet_101",
          "hrnet_w32", "dlcrnet_stride16_ms5"]


def best_snapshot(project: Path) -> Path | None:
    found = sorted(project.rglob("snapshot-best-*.pt"))
    return found[-1] if found else None


def settings_for(model: str, snapshot: Path, args) -> dict:
    sys.path.insert(0, str(REPO / "packages" / "cheese3d"))
    from cheese3d.backends.dlc import default_learning_rate

    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        # A warm start does not want the full initial rate: the weights are
        # already good and a large step undoes them before Adam settles.
        "learning_rate": default_learning_rate(model) * args.lr_scale,
        "save_every_n_epochs": args.eval_every,
        "validate_every_n_epochs": args.eval_every,
        "network_architecture": model,
        "training_shuffle": 1,
        "max_snapshots_to_keep": max(2, args.epochs // args.eval_every),
        "evaluation_split_file": str(args.split_file),
        "resume_from": str(snapshot),
        "key_metric": "test.rmse",
        "key_metric_asc": False,
        "dataloader_workers": args.workers,
        "display_iters": 20,
        "pin_memory": True,
        "persistent_workers": False,
        "rotation": 30, "scale_min": 0.5, "scale_max": 1.25,
        "crop_width": 448, "crop_height": 448,
        "motion_blur": False, "gaussian_noise": 12.75,
        "train_fraction_percent": 95,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-scale", type=float, default=0.1,
                        help="multiplier on the architecture's usual learning "
                             "rate, since this continues from trained weights "
                             "(default 0.1)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=int, default=60 * 60 * 40)
    parser.add_argument("--split-file",
                        default="/data/disk2/home/tony/cheese3d2_test_set/"
                                "cheese3d_demo_model-houlab-2025-05-28/selected_images.txt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source, dest = Path(args.source), Path(args.dest)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"{len(models)} model(s), {args.epochs} epochs each, selecting on test.rmse")
    print(f"source (untouched): {source}")
    print(f"dest:               {dest}\n")

    failures = []
    for position, model in enumerate(models, start=1):
        origin = source / f"dlc_{model}"
        snapshot = best_snapshot(origin)
        if snapshot is None:
            print(f"[{position}/{len(models)}] {model}: no snapshot, skipped")
            failures.append(model)
            continue
        print(f"[{position}/{len(models)}] {model}  resuming from {snapshot.name}",
              flush=True)
        if args.dry_run:
            continue

        project = dest / f"dlc_{model}"
        if not (project / "config.yaml").is_file():
            shutil.copytree(origin, project, symlinks=True)

        # The copy carries the old snapshot; resuming reads it by path, and a
        # fresh run would otherwise pick it up as an existing checkpoint.
        settings = settings_for(model, snapshot, args)
        logs = dest / "logs"
        logs.mkdir(exist_ok=True)
        log = logs / f"{model}.log"
        started = time.time()
        with log.open("w") as stream:
            # Own process group: DLC's dataloader workers ignore SIGTERM and
            # keep holding GPU memory after the parent exits.
            child = subprocess.Popen(
                ["timeout", str(args.timeout), "pixi", "run", "-e", "dlc",
                 "cheese3d", "--path", str(dest), "train", project.name,
                 "--gpu", args.gpu,
                 "--training-settings", json.dumps(settings)],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                cwd=REPO)
            code = child.wait()
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        status = "OK" if code == 0 else ("TIMEOUT" if code == 124 else "ERROR")
        detail = ""
        if status == "ERROR":
            text = log.read_text(errors="ignore")
            found = re.findall(r"[A-Za-z_.]*(?:Error|Exception)[^\n|]{0,110}", text)
            detail = found[-1].strip()[:110] if found else "see the log"
        print(f"  {status} in {(time.time() - started) / 3600:.2f} h"
              f"{'  ' + detail if detail else ''}", flush=True)
        if status != "OK":
            failures.append(model)

    print(f"\n{len(models) - len(failures)}/{len(models)} continued")
    if failures:
        print("problems: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
