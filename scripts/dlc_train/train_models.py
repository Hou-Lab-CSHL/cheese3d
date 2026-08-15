#!/usr/bin/env python
"""Train DeepLabCut architectures to completion, one project each.

Called by run_gpu0.sh and run_gpu1.sh, which pin this to a GPU and a set of
CPU cores. Every architecture gets its own Cheese3D project, is trained for
the full run, then stripped down and published to the results share.

Two things happen after each model finishes, because a finished project is
mostly weights and label copies that nobody needs again:

  * all snapshots except the best are deleted
  * the labeled-data folder is deleted -- it is a copy of the source
    annotations, and the source is untouched and still there

Projects are written straight to --output, which is local disk. A --share can
be given to copy the pruned result somewhere else afterwards; that copy
happens after pruning, so most of the bytes never cross the network. Training
directly onto a CIFS mount is not worth attempting: checkpoint writes there
are slow and the mount drops out under load.

Evaluation uses one fixed set of frames for every architecture, named by
--split-file, so the metrics are comparable between models rather than each
being scored on its own random holdout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "small_models"))
from common import (_RunWatcher, _kill_group, classify)  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TESTSET = REPO.parent / "cheese3d2_test_set"
DEFAULT_SOURCE = "cheese3d_demo_model-houlab-2025-05-28"
DEFAULT_OUTPUT = Path("/data/disk2/home/tony/dlc_projects")

# Measured on this data: 8 workers took 21% off epoch time against DLC's
# default of none, and more than 8 gave it back to per-epoch worker respawns.
DEFAULT_WORKERS = 8


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True,
                        help="comma-separated architectures to train in order")
    parser.add_argument("--gpu", default="0", help="GPU id for this script")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 picks a per-architecture batch size sized to "
                             "the card (see BATCH_SIZES)")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="epochs between evaluations, and between saved "
                             "snapshots -- the two are kept together so every "
                             "evaluated epoch has a checkpoint behind it "
                             "(default 10)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"directory holding one project per architecture "
                             f"(default {DEFAULT_OUTPUT})")
    parser.add_argument("--share", default="",
                        help="optional second location to copy finished, "
                             "pruned projects to, e.g. a mounted results "
                             "share. Omitted by default: --output is already "
                             "the result")
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET))
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="labelled DLC project inside --testset")
    parser.add_argument("--split-file", default="",
                        help="frames to evaluate on, one session/image.png per "
                             "line (default: selected_images.txt in --source)")
    parser.add_argument("--timeout", type=int, default=60 * 60 * 40,
                        help="per-model seconds before giving up (default 40 h)")
    parser.add_argument("--keep-labels", action="store_true",
                        help="do not delete labeled-data after training")
    parser.add_argument("--keep-snapshots", action="store_true",
                        help="do not delete all but the best snapshot")
    parser.add_argument("--no-publish", action="store_true",
                        help="skip the --share copy even when one is given")
    return parser.parse_args(argv)


# Peak memory at batch 32 was 19.8 GiB for hrnet_w18 on a 48 GB card. These
# scale that by backbone size so a single 80 GB card is used without being
# overrun; the larger HRNets and DEKRs hold far more activation per sample.
BATCH_SIZES = {
    "hrnet_w18": 32, "dekr_w18": 32,
    "resnet_50": 32, "resnet_101": 24,
    "hrnet_w32": 24, "dekr_w32": 24,
    "hrnet_w48": 16, "dekr_w48": 16,
    "dlcrnet_stride32_ms5": 24, "dlcrnet_stride16_ms5": 16,
}


def batch_for(model: str, override: int) -> int:
    return override if override > 0 else BATCH_SIZES.get(model, 16)


def build_project(staging: Path, project: str, testset: Path, source: str) -> Path:
    """Create a Cheese3D project seeded from the labelled DLC source."""
    directory = staging / project
    if (directory / "config.yaml").is_file():
        print(f"  reusing existing project at {directory}", flush=True)
        return directory
    staging.mkdir(parents=True, exist_ok=True)

    base = ["pixi", "run", "-e", "dlc", "cheese3d", "--path", str(staging)]
    subprocess.run(base + ["setup", project], check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import yaml
    config = directory / "config.yaml"
    cfg = yaml.safe_load(config.read_text())
    cfg["model"]["name"] = f"{project}_model"
    cfg["model"]["backend_type"] = "dlc"
    cfg["model"]["backend_options"] = {
        "source_project_path": str(testset / source),
        "source_format": "dlc",
    }
    cfg["sessions"] = [{"name": "20231031_chew"}]
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    videos = directory / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    origin = testset / "test_vid" / "20231031_chew"
    destination = videos / origin.name
    if origin.is_dir() and not destination.exists():
        try:
            # Hardlinks are instant and cost no disk, but only work within one
            # filesystem -- and cp creates the directory before it discovers
            # that, so the partial result has to go before falling back.
            subprocess.run(["cp", "-al", str(origin), str(videos)], check=True,
                           stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(origin, destination)

    subprocess.run(base + ["summarize", project], check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return directory


def settings_for(model: str, args: argparse.Namespace, split_file: Path) -> dict:
    sys.path.insert(0, str(REPO / "packages" / "cheese3d"))
    from cheese3d.backends.dlc import default_learning_rate

    return {
        "epochs": args.epochs,
        "batch_size": batch_for(model, args.batch_size),
        # HRNet, DEKR and DLCRNet freeze pretrained BatchNorm statistics and
        # will not train at ResNet's 5e-4; hrnet_w32 sat at a flat loss until
        # it was dropped to 1e-4.
        "learning_rate": default_learning_rate(model),
        "save_every_n_epochs": args.eval_every,
        "validate_every_n_epochs": args.eval_every,
        "network_architecture": model,
        "training_shuffle": 1,
        # Snapshots are pruned to the best one afterwards, but DLC has to keep
        # enough of them during the run for that best one to still exist.
        "max_snapshots_to_keep": max(2, args.epochs // args.eval_every),
        "evaluation_split_file": str(split_file),
        "dataloader_workers": args.workers,
        "pin_memory": True,
        # No measured benefit on this data, and slower at low worker counts.
        "persistent_workers": False,
        "rotation": 30, "scale_min": 0.5, "scale_max": 1.25,
        "crop_width": 448, "crop_height": 448,
        "motion_blur": False, "gaussian_noise": 12.75,
        # train_fraction_percent only labels the output folder now that the
        # split is fixed; keep it truthful about the split actually used.
        "train_fraction_percent": 95,
    }


def train_one(model: str, directory: Path, staging: Path, args: argparse.Namespace,
              split_file: Path) -> "object":
    """Run one architecture to completion and return its outcome."""
    logs = staging / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{model}.log"
    settings = settings_for(model, args, split_file)

    print(f"  batch {settings['batch_size']}, lr {settings['learning_rate']}, "
          f"{args.epochs} epochs, eval every {args.eval_every}", flush=True)
    started = time.time()
    with log.open("w") as stream:
        child = subprocess.Popen(
            ["timeout", str(args.timeout), "pixi", "run", "-e", "dlc",
             "cheese3d", "--path", str(staging), "train", directory.name,
             "--gpu", args.gpu, "--training-settings", json.dumps(settings)],
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            cwd=REPO,
        )
        with _RunWatcher(child.pid, log) as watcher:
            code = child.wait()
    elapsed = time.time() - started
    outcome = classify(log, code, args.timeout)
    outcome.model, outcome.seconds = model, elapsed
    outcome.peak_mib = watcher.peak
    outcome.batch_size = settings["batch_size"]
    outcome.epoch_seconds = watcher.epoch_seconds
    outcome.utilization = watcher.mean_utilization
    _kill_group(child.pid)
    return outcome


def prune_snapshots(directory: Path) -> int:
    """Delete every snapshot but the best, returning bytes reclaimed.

    DLC names the best one `snapshot-best-*`; the rest are per-epoch
    checkpoints it no longer needs once training has finished.
    """
    freed = 0
    for train_dir in directory.rglob("dlc-models-pytorch/**/train"):
        snapshots = sorted(train_dir.glob("snapshot-*.pt"))
        best = [s for s in snapshots if "best" in s.name]
        keep = set(best) or set(snapshots[-1:])   # fall back to the last one
        for snapshot in snapshots:
            if snapshot not in keep:
                freed += snapshot.stat().st_size
                snapshot.unlink()
    return freed


def drop_labeled_data(directory: Path) -> int:
    """Delete the copied annotations, returning bytes reclaimed."""
    freed = 0
    for folder in directory.rglob("labeled-data"):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                freed += path.stat().st_size
        shutil.rmtree(folder)
    return freed


def publish(directory: Path, share: Path) -> Optional[Path]:
    """Copy a finished project to the results share."""
    target = share / directory.name
    try:
        share.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(directory, target, symlinks=True)
        return target
    except (OSError, shutil.Error) as error:
        print(f"  could not publish to {target}: {error}", flush=True)
        return None


def human(size: int) -> str:
    return f"{size / 1024 ** 3:.1f} GiB"


def main() -> int:
    args = parse_args()
    staging = Path(args.output)
    testset = Path(args.testset)
    split_file = (Path(args.split_file) if args.split_file
                  else testset / args.source / "selected_images.txt")
    if not split_file.is_file():
        sys.exit(f"evaluation split file not found: {split_file}")
    share = Path(args.share) if args.share else None

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"GPU {args.gpu}: {len(models)} models, {args.epochs} epochs each")
    print(f"cores:  {sorted(os.sched_getaffinity(0))}")
    print(f"output: {staging}")
    if share:
        print(f"copy:   {share}")
    print(f"split:  {split_file}\n", flush=True)

    failures = []
    for position, model in enumerate(models, start=1):
        print(f"[{position}/{len(models)}] {model}", flush=True)
        project = f"dlc_{model}"
        try:
            directory = build_project(staging, project, testset, args.source)
        except subprocess.CalledProcessError as error:
            print(f"  project setup failed: {error}", flush=True)
            failures.append(model)
            continue

        outcome = train_one(model, directory, staging, args, split_file)
        print(f"  {outcome.status} in {outcome.seconds / 3600:.2f} h"
              f"{'' if not outcome.detail else '  ' + outcome.detail}", flush=True)
        if outcome.status != "OK":
            failures.append(model)
            continue

        if not args.keep_snapshots:
            print(f"  pruned snapshots: {human(prune_snapshots(directory))}", flush=True)
        if not args.keep_labels:
            print(f"  dropped labeled-data: {human(drop_labeled_data(directory))}",
                  flush=True)
        print(f"  kept: {directory}", flush=True)
        if share and not args.no_publish:
            target = publish(directory, share)
            if target:
                print(f"  copied to {target}", flush=True)
            else:
                failures.append(f"{model} (copy)")

    print(f"\nGPU {args.gpu} finished. "
          f"{len(models) - len([f for f in failures if '(' not in f])}"
          f"/{len(models)} trained.")
    if failures:
        print("problems: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
