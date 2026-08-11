from pathlib import Path

import pytest

from cheese3d.project import resolve_pose3d_csv


def test_resolve_pose3d_csv_accepts_anipose_scorer_suffix(tmp_path):
    pose3d_dir = tmp_path / "pose-3d"
    pose3d_dir.mkdir()
    output = pose3d_dir / "recording_DLC_resnet50_modelshuffle1_100.csv"
    output.touch()

    assert resolve_pose3d_csv(tmp_path) == output


def test_resolve_pose3d_csv_reports_missing_triangulation(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run triangulation"):
        resolve_pose3d_csv(tmp_path)


def test_resolve_pose3d_csv_rejects_ambiguous_outputs(tmp_path):
    pose3d_dir = tmp_path / "pose-3d"
    pose3d_dir.mkdir()
    (pose3d_dir / "recording_model_a.csv").touch()
    (pose3d_dir / "recording_model_b.csv").touch()

    with pytest.raises(RuntimeError, match="Multiple 3-D pose CSV"):
        resolve_pose3d_csv(tmp_path)
