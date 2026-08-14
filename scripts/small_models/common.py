"""Shared driver for the small-model smoke tests.

One Python module per backend sits beside this file, because the three
backends cannot share a Python process -- the Lightning Pose environment has
no DeepLabCut installed and vice versa. Each is run under its own Pixi
environment by ``run.sh``; this module holds everything that does not depend
on which backend is active.

The test trains each architecture for a small number of epochs and reports
whether it ran, not whether it learned anything. One epoch is enough to prove
a model builds, accepts this project's data, and takes optimizer steps; raise
``--epochs`` when you want a signal about training quality as well.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class Result:
    model: str
    status: str          # OK | OOM | ERROR | TIMEOUT
    seconds: float
    detail: str = ""
    peak_mib: int = 0    # highest memory this run held on any single GPU
    batch_size: int = 0
    epoch_seconds: Optional[float] = None   # cost of one epoch, startup excluded
    overhead_seconds: float = 0.0           # setup + final evaluation
    per_sample_mib: Optional[float] = None  # activation cost of one sample
    fixed_mib: Optional[float] = None       # weights, optimizer, CUDA context


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the options every backend script accepts."""
    parser.add_argument("--epochs", type=int, default=1,
                        help="epochs per model (default 1: a build/step check only)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="samples per batch (default 16). SLEAP counts this "
                             "per GPU under DDP; DLC and Lightning Pose count it "
                             "across the run")
    parser.add_argument("--gpu", default="0",
                        help="comma-separated GPU ids, e.g. 0 or 0,1")
    parser.add_argument("--project-epochs", type=int, default=300,
                        help="epoch count the report projects run time to "
                             "(default 300)")
    parser.add_argument("--target-gb", type=int, default=90,
                        help="card size the report extrapolates batch sizes to "
                             "(default 90, an H100 NVL)")
    parser.add_argument("--fit-batch", action="store_true",
                        help="also run each model at half the batch size, so "
                             "the maximum batch is fitted rather than bounded. "
                             "Roughly doubles the run time")
    parser.add_argument("--models", default="",
                        help="comma-separated subset of the backend's small models")
    parser.add_argument("--root", default="",
                        help="where to build the throwaway project "
                             "(default: <repo>/../cheese3d_small_model_tests)")
    parser.add_argument("--report-name", default="SMALL_MODEL_RESULTS.md",
                        help="report filename inside --report-dir")
    parser.add_argument("--report-dir", default="",
                        help="where to write the report (default: "
                             "<repo>/reports/small_models). Kept apart from "
                             "--root so results survive deleting the projects")
    parser.add_argument("--testset", default="",
                        help="directory holding the labelled source project and "
                             "test videos (default: <repo>/../cheese3d2_test_set)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-model timeout in seconds")
    parser.add_argument("--keep", action="store_true",
                        help="reuse an existing project instead of rebuilding")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_paths(args) -> tuple[Path, Path]:
    """Return (project root, testset), defaulting relative to the repository."""
    parent = repo_root().parent
    root = Path(args.root) if args.root else parent / "cheese3d_small_model_tests"
    testset = Path(args.testset) if args.testset else parent / "cheese3d2_test_set"
    if not testset.is_dir():
        sys.exit(f"testset not found: {testset} (pass --testset)")
    return root, testset


def resolve_report_dir(args, suite: str = "small_models") -> Path:
    """Where the report lives -- inside the repo, not beside the projects.

    build_project deletes and recreates its project tree on every run, so a
    report kept there would be thrown away with it. Precedence runs
    --report-dir, then $CHEESE3D_REPORT_DIR, then the repo default: the
    environment variable is what lets one setting cover every suite at once,
    and the reports do not collide because each writes its own filename.
    """
    import os

    if args.report_dir:
        return Path(args.report_dir)
    chosen = os.environ.get("CHEESE3D_REPORT_DIR", "").strip()
    if chosen:
        return Path(chosen).expanduser()
    return repo_root() / "reports" / suite


def _suite_of(report_name: str) -> str:
    """Which suite a report belongs to, so a direct python run still lands right.

    The runner scripts always pass --report-dir, so this only matters when a
    backend script is invoked on its own.
    """
    if report_name.startswith("MEDIUM"):
        return "medium_models"
    if report_name.startswith("MULTI_GPU"):
        return "multi_gpu"
    return "small_models"


def build_project(root: Path, project: str, backend_type: str, env: str,
                  testset: Path, keep: bool) -> None:
    """Create a Cheese3D project seeded from the labelled DLC source."""
    directory = root / project
    if keep and (directory / "config.yaml").is_file():
        print(f"reusing existing project at {directory}")
        return
    if directory.exists():
        shutil.rmtree(directory)
    root.mkdir(parents=True, exist_ok=True)

    run = ["pixi", "run", "-e", env, "cheese3d", "--path", str(root)]
    _quiet(run + ["setup", project])

    config = directory / "config.yaml"
    import yaml  # available in every backend environment
    cfg = yaml.safe_load(config.read_text())
    cfg["model"]["name"] = f"{project}_model"
    cfg["model"]["backend_type"] = backend_type
    cfg["model"]["backend_options"] = {
        "source_project_path": str(testset / "cheese3d_demo_model-houlab-2025-05-28"),
        "source_format": "dlc",
    }
    # find_videos only scans the sessions named here; the default empty list
    # silently discovers nothing.
    cfg["sessions"] = [{"name": "20231031_chew"}]
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    videos = directory / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    source = testset / "test_vid" / "20231031_chew"
    destination = videos / source.name
    if not destination.exists():
        try:                      # hardlink when possible: instant, no extra disk
            subprocess.run(["cp", "-al", str(source), str(videos)], check=True)
        except subprocess.CalledProcessError:
            shutil.copytree(source, destination)

    _quiet(run + ["summarize", project])


def _quiet(command: List[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def classify(log: Path, returncode: int, timeout: int) -> Result:
    """Turn a finished run into a status plus a one-line reason."""
    text = log.read_text(errors="ignore") if log.is_file() else ""
    if "OutOfMemoryError" in text:
        size = _search(text, r"Tried to allocate [0-9.]+ [GMK]iB")
        return Result("", "OOM", 0.0, size)
    if returncode == 124:
        return Result("", "TIMEOUT", 0.0, f"exceeded {timeout}s")
    if returncode != 0:
        return Result("", "ERROR", 0.0, _last_error(text))
    return Result("", "OK", 0.0)


def _search(text: str, pattern: str) -> str:
    import re
    found = re.findall(pattern, text)
    return found[-1] if found else ""


def _last_error(text: str) -> str:
    import re
    found = re.findall(r"[A-Za-z_.]*(?:Error|Exception)[^\n|│]{0,110}", text)
    found = [f.strip() for f in found if "OutOfMemory" not in f]
    return found[-1][:110] if found else "no error text captured"


def run_models(models: List[str], settings_for: Callable[[str], Dict],
               root: Path, project: str, env: str, args) -> List[Result]:
    """Train each model in turn and collect the outcomes."""
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results: List[Result] = []
    projection = getattr(args, "project_epochs", 300)
    print(f"  {'model':24s} {'result':8s} {'time':>6s} {'peak GPU':>9s} "
          f"{f'{projection}ep':>9s}")

    def one_run(model: str, batch: int, suffix: str = "") -> Result:
        """Train one model at one batch size and measure what it cost."""
        log = logs / f"{env}_{model}{suffix}.log"
        original = args.batch_size
        args.batch_size = batch          # settings_for reads the batch off args
        try:
            settings = settings_for(model)
        finally:
            args.batch_size = original
        started = time.time()
        with log.open("w") as stream:
            # Own process group: SLEAP's DDP workers ignore SIGTERM and keep
            # holding GPU memory after the parent exits, so they have to be
            # killed as a group. Matching on the interpreter path instead
            # would also match this script, which runs under the very same
            # environment -- it would kill itself.
            child = subprocess.Popen(
                ["timeout", str(args.timeout), "pixi", "run", "-e", env,
                 "cheese3d", "--path", str(root), "train", project,
                 "--gpu", args.gpu, "--training-settings", json.dumps(settings)],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
            # start_new_session put the child in its own group, whose id is its
            # pid -- that is what the watcher attributes GPU memory to.
            with _RunWatcher(child.pid, log) as watcher:
                returncode = child.wait()
        elapsed = time.time() - started
        outcome = classify(log, returncode, args.timeout)
        outcome.model, outcome.seconds = model, elapsed
        outcome.peak_mib = watcher.peak
        outcome.batch_size = batch
        outcome.epoch_seconds = watcher.epoch_seconds
        if outcome.epoch_seconds:
            outcome.overhead_seconds = max(
                0.0, elapsed - outcome.epoch_seconds * args.epochs)
        _kill_group(child.pid)
        time.sleep(5)
        return outcome

    for model in models:
        print(f"  {model:24s} ", end="", flush=True)
        result = one_run(model, args.batch_size)
        memory = f"{result.peak_mib / 1024:5.1f} GiB" if result.peak_mib else "    -    "
        projected = _format_duration(_projected_seconds(result, projection))
        print(f"{result.status:8s} {result.seconds:6.0f}s {memory} "
              f"{projected:>9s}  {result.detail}")
        _fit_memory(result, one_run, args)
        results.append(result)
    return results


def _fit_memory(result: Result, one_run: Callable[..., Result], args) -> None:
    """Split peak memory into fixed and per-sample cost, via a second run.

    One measurement cannot tell the two apart: a model's peak is its weights,
    optimizer state and CUDA context -- none of which grow with batch -- plus
    activations, which do. Running the same model at half the batch and taking
    the difference cancels the fixed part, which turns a lower bound on the
    usable batch size into an actual estimate.
    """
    batch = result.batch_size
    if not getattr(args, "fit_batch", False) or result.status != "OK":
        return
    if batch < 4 or not result.peak_mib:
        return          # too small to halve into two distinguishable points

    half = batch // 2
    print(f"  {'  fitting at batch ' + str(half):24s} ", end="", flush=True)
    other = one_run(result.model, half, suffix="_fit")
    print(f"{other.status:8s} {other.seconds:6.0f}s "
          f"{other.peak_mib / 1024:5.1f} GiB" if other.peak_mib
          else f"{other.status:8s}")

    if other.status != "OK" or not other.peak_mib:
        return
    per_sample = (result.peak_mib - other.peak_mib) / (batch - half)
    if per_sample <= 0:
        # Below the point where activations dominate, noise can swamp the
        # difference. Reporting the floor is honest; inventing a slope is not.
        return
    result.per_sample_mib = per_sample
    result.fixed_mib = max(0.0, result.peak_mib - per_sample * batch)


def _gpu_index_by_uuid() -> Dict[str, str]:
    """Map each GPU's UUID to its index, since compute-apps reports UUIDs."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    mapping = {}
    for line in out.strip().splitlines():
        uuid, _, index = line.partition(",")
        mapping[uuid.strip()] = index.strip()
    return mapping


def _sample_group_memory(pgid: int, by_uuid: Dict[str, str]) -> Dict[str, int]:
    """Return MiB in use per GPU by the processes of one process group.

    Reading the device total instead would pick up whatever else shares the
    card; attributing memory to the run's own process group is what makes the
    number comparable between models.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    import os
    per_gpu: Dict[str, int] = {}
    for line in out.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 3:
            continue
        uuid, pid, mib = fields
        try:
            if os.getpgid(int(pid)) != pgid:
                continue
            used = int(mib)
        except (ValueError, ProcessLookupError, PermissionError):
            continue    # process exited, or belongs to another user
        index = by_uuid.get(uuid, uuid)
        per_gpu[index] = per_gpu.get(index, 0) + used
    return per_gpu


# DeepLabCut prints "Epoch 3/10 (lr=0.0001), train loss ..." once an epoch is
# done; Lightning (SLEAP and Lightning Pose) redraws "Epoch 2: 100%|..." many
# times over. Either way the first sighting of an index means that epoch ended.
_EPOCH_DONE = re.compile(r"Epoch (?P<dlc>\d+)/\d+ \(lr=|"
                         r"Epoch (?P<lightning>\d+): 100%")
# Lightning's progress bar carries its own elapsed time: "[01:12<00:00, ...]".
_LIGHTNING_ELAPSED = re.compile(r"Epoch \d+: 100%.*?\[(?P<m>\d+):(?P<s>\d+)<")


class _RunWatcher:
    """Follow a run in the background, keeping peak GPU memory and epoch times.

    Peak memory matters rather than average: a batch size fits or it does not,
    and what decides that is the highest point the run ever reaches. Epoch
    boundaries are timestamped from the log because a 300-epoch projection
    needs the cost of an epoch alone, without the startup wrapped around it.
    """

    def __init__(self, pgid: int, log: Path, interval: float = 1.5) -> None:
        import threading
        self._pgid = pgid
        self._log = log
        self._interval = interval
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._offset = 0
        self.peak_per_gpu: Dict[str, int] = {}
        self.epoch_seen: Dict[int, float] = {}   # epoch index -> time first seen
        self.reported_epoch_seconds: Optional[float] = None

    def _loop(self) -> None:
        by_uuid = _gpu_index_by_uuid()
        while not self._done.is_set():
            for index, mib in _sample_group_memory(self._pgid, by_uuid).items():
                if mib > self.peak_per_gpu.get(index, 0):
                    self.peak_per_gpu[index] = mib
            self._read_log()
            self._done.wait(self._interval)
        self._read_log()      # catch whatever landed after the final sample

    def _read_log(self) -> None:
        """Timestamp epoch boundaries as they appear in the training log."""
        try:
            with self._log.open("rb") as stream:
                stream.seek(self._offset)
                chunk = stream.read()
                self._offset = stream.tell()
        except OSError:
            return
        if not chunk:
            return
        text = chunk.decode("utf-8", errors="ignore")
        now = time.time()
        for match in _EPOCH_DONE.finditer(text):
            # Lightning redraws its bar many times per epoch; only the first
            # sighting of an epoch index marks when that epoch finished.
            index = int(match.group("dlc") or match.group("lightning"))
            self.epoch_seen.setdefault(index, now)
        # Lightning also prints each epoch's own duration, which is the only
        # per-epoch timing available when a run is a single epoch long.
        for match in _LIGHTNING_ELAPSED.finditer(text):
            self.reported_epoch_seconds = (int(match.group("m")) * 60
                                           + int(match.group("s")))

    def __enter__(self) -> "_RunWatcher":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._done.set()
        self._thread.join(timeout=10)

    @property
    def peak(self) -> int:
        """Highest MiB held on any one GPU -- the number that sets the ceiling."""
        return max(self.peak_per_gpu.values(), default=0)

    @property
    def epoch_seconds(self) -> Optional[float]:
        """Seconds per epoch, excluding startup and final evaluation.

        Two or more epochs give this directly, for any backend, as the gap
        between consecutive epoch boundaries. A single-epoch run cannot
        separate the epoch from the startup around it, so fall back to the
        duration Lightning reports for itself; DeepLabCut prints no such
        figure, and there the answer is simply unknown.
        """
        stamps = [self.epoch_seen[i] for i in sorted(self.epoch_seen)]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        if gaps:
            gaps.sort()
            return gaps[len(gaps) // 2]      # median: ignore a slow first epoch
        return self.reported_epoch_seconds


def _kill_group(pid: int) -> None:
    """Kill a finished run's process group, leaving this script alone."""
    import os
    import signal

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def report(backend: str, results: List[Result], args) -> int:
    """Print a summary, write it into the shared report, return an exit code."""
    out_dir = resolve_report_dir(args, _suite_of(getattr(args, "report_name", "")))
    ok = [r for r in results if r.status == "OK"]
    print(f"\n{backend}: {len(ok)}/{len(results)} ran at {args.epochs} epoch(s)")
    for r in results:
        if r.status != "OK":
            print(f"  {r.status:8s} {r.model:24s} {r.detail}")

    target = getattr(args, "target_gb", 0)
    batch = getattr(args, "batch_size", 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    projection = getattr(args, "project_epochs", 300)
    lines = [f"\n## {backend} ({args.epochs} epoch(s), batch {batch}, GPU {args.gpu})\n",
             "| model | result | time | peak GPU | s/epoch | "
             f"est. {projection} epochs | est. batch @ {target} GB | detail |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    total = 0.0
    for r in results:
        memory = f"{r.peak_mib / 1024:.1f} GiB" if r.peak_mib else "–"
        per_epoch = f"{r.epoch_seconds:.0f}s" if r.epoch_seconds else "–"
        full = _projected_seconds(r, projection)
        total += full or 0.0
        lines.append(f"| `{r.model}` | {r.status} | {r.seconds:.0f}s | {memory} | "
                     f"{per_epoch} | {_format_duration(full)} | "
                     f"{_estimated_batch(r, target)} | {r.detail or '–'} |")
    if total:
        lines.append(f"\n**{backend} total, {projection} epochs, GPU "
                     f"{args.gpu}, run back to back: "
                     f"{_format_duration(total)}**")
    lines += ["", _PROJECTION_NOTE.format(epochs=projection),
              "", _EXTRAPOLATION_NOTE.format(target=target)]
    name = getattr(args, "report_name", "") or "SMALL_MODEL_RESULTS.md"
    _write_section(out_dir / name, backend, "\n".join(lines) + "\n")
    print(f"report: {out_dir / name}")
    return 0 if len(ok) == len(results) else 1


_PROJECTION_NOTE = """\
The {epochs}-epoch estimate is startup plus per-epoch cost times {epochs}, not
the measured run time multiplied out -- at one epoch most of the wall clock is
setup and final evaluation, so multiplying the total would be badly wrong.

Seconds per epoch comes from the gap between epoch boundaries in the log,
which needs at least two epochs to measure. With a single epoch, SLEAP and
Lightning Pose still report their own epoch duration and are timed from that,
but DeepLabCut prints no such figure and shows `–`. Run with `--epochs 3` to
time every backend properly.

The estimate assumes epochs cost the same throughout, which holds for fixed
schedules; it does not model early stopping or a changing batch size.\
"""

_EXTRAPOLATION_NOTE = """\
Peak GPU is the highest memory the run held on a single card, sampled from
`nvidia-smi` and attributed to the run's own process group.

Batch estimates come in two forms, and the prefix says which:

`~ N` is fitted. The model was run a second time at half the batch, and the
difference in peak memory gives the per-sample cost with the fixed cost
cancelled out. The estimate is then `({target} GB - fixed) / per-sample`. This
is a real maximum, though leave a margin: memory also fragments, and the last
batch that fits in a probe can still fail once a run is under way.

`>= N` is only a lower bound, produced when `--fit-batch` was not used. It
divides peak by batch, which charges the fixed cost (CUDA context, weights,
optimizer state -- none of which grow with batch) to every sample. The true
maximum is higher, and the gap is widest for the models measured at the
smallest batches.

Extrapolating across GPU generations adds error on top of that -- attention
and convolution kernels pick different algorithms and workspace sizes per
architecture, so a figure measured on one card is a guide on another, not a
guarantee.\
"""


def _projected_seconds(result: Result, epochs: int) -> Optional[float]:
    """Wall-clock for a full training run: startup, N epochs, evaluation."""
    if result.status != "OK" or not result.epoch_seconds:
        return None
    return result.overhead_seconds + result.epoch_seconds * epochs


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    hours, minutes = divmod(int(seconds) // 60, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def _estimated_batch(result: Result, target_gb: int) -> str:
    """Largest batch expected to fit a target card.

    With a two-run fit this is a real estimate, marked `~`. Without one it is
    only a lower bound, marked `>=`, because the fixed cost gets charged to
    every sample.
    """
    if result.status != "OK" or not result.peak_mib or not result.batch_size:
        return "–"
    budget = target_gb * 1024
    if result.per_sample_mib and result.fixed_mib is not None:
        usable = budget - result.fixed_mib
        if usable <= 0:
            return "0 (does not fit)"
        return f"~ {int(usable / result.per_sample_mib)}"
    return f"≥ {int(budget / (result.peak_mib / result.batch_size))}"


def _write_section(path: Path, backend: str, section: str) -> None:
    """Replace this backend's section, leaving the other backends' intact.

    Rerunning one backend has to refresh its own rows without disturbing the
    rest of the report, and without stacking a second copy of them underneath.
    """
    kept: List[str] = []
    if path.is_file():
        skipping = False
        for line in path.read_text().splitlines(keepends=True):
            if line.startswith("## "):
                skipping = line.startswith(f"## {backend} (")
            if not skipping:
                kept.append(line)
    path.write_text("".join(kept).rstrip("\n") + "\n" + section)
