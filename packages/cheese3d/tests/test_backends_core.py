"""CPU-only tests for shared pose-backend utilities."""
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cheese3d.backends.core import (
    active_pose_backend,
    isolate_worker_output,
    read_dlc_records,
    read_foreign_records,
    read_lp_records,
    write_dlc_records,
    write_lp_records,
)


def test_isolate_worker_output_detaches_fd_1_and_2_from_the_inherited_pipe(tmp_path):
    """multiprocessing.get_context("spawn") workers inherit fd 1/2 directly

    from their parent. Under Textual Serve, that fd is the pipe textual-serve
    uses to send rendered frames to the browser; several worker processes
    (one per camera/GPU) writing to it concurrently corrupts that framing
    protocol and silently breaks the browser's connection with no way to
    recover -- confirmed as the actual cause of a real reported freeze during
    video generation/tracking that was otherwise indistinguishable from a
    Python deadlock. This spawns a real child process, has it call
    isolate_worker_output(), then write to stdout/stderr via both the Python
    layer and a raw os.write(1, ...) (mimicking a C-level library write) --
    none of it should reach the parent's pipe.
    """
    import os
    import subprocess
    import sys

    script = tmp_path / "worker.py"
    script.write_text(
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from cheese3d.backends.core import isolate_worker_output\n"
        "isolate_worker_output()\n"
        "print('python-level stdout leak')\n"
        "print('python-level stderr leak', file=sys.stderr)\n"
        "import os\n"
        "os.write(1, b'raw fd 1 leak')\n"
        % str(Path(__file__).resolve().parents[1])
    )

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def _fake_version(installed):
    def version(distribution):
        if distribution in installed:
            return "0.0.0"
        raise importlib.metadata.PackageNotFoundError(distribution)
    return version


def test_active_pose_backend_detects_deeplabcut(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _fake_version({"deeplabcut"}))

    assert active_pose_backend() == "dlc"


def test_active_pose_backend_detects_lightning_pose(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _fake_version({"lightning-pose"}))

    assert active_pose_backend() == "lightning_pose"


def test_active_pose_backend_detects_sleap(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _fake_version({"sleap"}))

    assert active_pose_backend() == "sleap"


def test_active_pose_backend_returns_none_when_nothing_installed(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _fake_version(set()))

    assert active_pose_backend() is None


def test_read_dlc_records_matches_normalized_record_shape(tmp_path):
    label_dir = tmp_path / "labeled-data" / "cam_L"
    label_dir.mkdir(parents=True)
    (label_dir / "img1.png").write_bytes(b"image")
    columns = pd.MultiIndex.from_tuples([
        ("tester", bodypart, coord) for bodypart in ["nose", "ear"] for coord in ("x", "y")
    ])
    index = pd.MultiIndex.from_tuples([("labeled-data", "cam_L", "img1.png")])
    pd.DataFrame([[4.0, 5.0, 8.0, 9.0]], index=index, columns=columns).to_hdf(
        label_dir / "CollectedData_tester.h5", key="df"
    )

    records = read_dlc_records(tmp_path, ["nose", "ear"])

    assert set(records.keys()) == {("cam_L", "img1.png")}
    image_path, points = records[("cam_L", "img1.png")]
    assert image_path == label_dir / "img1.png"
    assert points == [[4.0, 5.0], [8.0, 9.0]]


def test_read_dlc_records_fills_missing_keypoints_with_nan(tmp_path):
    """A keypoint absent from this folder's schema must not crash the read."""
    label_dir = tmp_path / "labeled-data" / "cam_R"
    label_dir.mkdir(parents=True)
    (label_dir / "img2.png").write_bytes(b"image")
    columns = pd.MultiIndex.from_tuples([("tester", "nose", coord) for coord in ("x", "y")])
    index = pd.MultiIndex.from_tuples([("labeled-data", "cam_R", "img2.png")])
    pd.DataFrame([[1.0, 2.0]], index=index, columns=columns).to_hdf(
        label_dir / "CollectedData_tester.h5", key="df"
    )

    records = read_dlc_records(tmp_path, ["nose", "ear"])

    _, points = records[("cam_R", "img2.png")]
    assert points[0] == [1.0, 2.0]
    assert all(np.isnan(v) for v in points[1])


def test_read_lp_records_resolves_images_relative_to_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    label_dir = data_dir / "labeled-data" / "cam_L"
    label_dir.mkdir(parents=True)
    (label_dir / "img1.png").write_bytes(b"image")
    columns = pd.MultiIndex.from_tuples([
        ("lightning_pose", bodypart, coord)
        for bodypart in ["nose", "ear"] for coord in ("x", "y")
    ], names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame([[4.0, 5.0, np.nan, np.nan]],
                      index=["labeled-data/cam_L/img1.png"], columns=columns)
    df.to_csv(data_dir / "CollectedData.csv")

    records = read_lp_records(tmp_path, ["nose", "ear"])

    assert set(records.keys()) == {("cam_L", "img1.png")}
    image_path, points = records[("cam_L", "img1.png")]
    assert image_path == label_dir / "img1.png"
    assert points[0] == [4.0, 5.0]
    assert all(np.isnan(v) for v in points[1])


def test_read_lp_records_returns_empty_when_no_collected_data(tmp_path):
    assert read_lp_records(tmp_path, ["nose"]) == {}


def test_write_dlc_records_produces_dlc_readable_layout(tmp_path):
    """A round trip through write then read must preserve points and images."""
    image = tmp_path / "source" / "img1.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    records = {("cam_L", "img1.png"): (image, [[4.0, 5.0], [float("nan"), float("nan")]])}
    dlc_project = tmp_path / "dlc"

    summary = write_dlc_records(records, dlc_project, "tester", ["nose", "ear"])

    assert summary == {"folders": 1, "images": 1, "records": 1}
    label_dir = dlc_project / "labeled-data" / "cam_L"
    assert (label_dir / "img1.png").is_file()
    assert (label_dir / "CollectedData_tester.h5").is_file()
    assert (label_dir / "CollectedData_tester.csv").is_file()

    read_back = read_dlc_records(dlc_project, ["nose", "ear"])
    _, points = read_back[("cam_L", "img1.png")]
    assert points[0] == [4.0, 5.0]
    assert np.isnan(points[1]).all()


def test_write_lp_records_produces_lp_readable_layout(tmp_path):
    image = tmp_path / "source" / "img1.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    records = {("cam_L", "img1.png"): (image, [[4.0, 5.0]])}
    lp_project = tmp_path / "lp"

    summary = write_lp_records(records, lp_project, ["nose"])

    assert summary["num_frames"] == 1
    assert summary["copied_images"] == 1
    destination = lp_project / "data" / "labeled-data" / "cam_L" / "img1.png"
    assert destination.is_file()

    read_back = read_lp_records(lp_project, ["nose"])
    assert read_back[("cam_L", "img1.png")][1] == [[4.0, 5.0]]


def test_read_foreign_records_dispatches_by_source_format(tmp_path):
    assert read_foreign_records("dlc", tmp_path, ["nose"]) == {}
    assert read_foreign_records("lightning_pose", tmp_path, ["nose"]) == {}
    with pytest.raises(ValueError):
        read_foreign_records("unknown", tmp_path, ["nose"])
