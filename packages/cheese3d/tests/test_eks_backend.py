from pathlib import Path

import pandas as pd
import pytest

from cheese3d.backends.core import register_pose_backend
from cheese3d.backends.eks import EKSBackend
from cheese3d.config import KeypointConfig

class FakePrimitiveBackend:
    def __init__(self, name, root_dir, videos, keypoints, crops=None, value=None):
        self.name = name
        self.root_dir = Path(root_dir)
        self.videos = videos
        self.keypoints = keypoints
        self.crops = crops
        self.value = value

    @property
    def project_path(self):
        return self.root_dir

@pytest.fixture(autouse=True)
def register_fake_backend():
    register_pose_backend("fake_primitive", FakePrimitiveBackend)

@pytest.mark.unit
def test_eks_backend_builds_submodels_at_nested_roots(tmp_path):
    backend = EKSBackend(name="ensemble",
                         root_dir=tmp_path / "model" / "ensemble" / "backend",
                         videos=[tmp_path / "video.avi"],
                         keypoints=[KeypointConfig(label="nose")],
                         models={
                             "rng0": {"backend_type": "fake_primitive",
                                      "backend_options": {"value": 0}},
                             "rng1": {"backend_type": "fake_primitive",
                                      "backend_options": {"value": 1}},
                         })
    assert backend.models["rng0"].project_path == backend.project_path / "rng0" / "backend"
    assert backend.models["rng1"].project_path == backend.project_path / "rng1" / "backend"
    assert backend.models["rng0"].value == 0
    assert backend.ensemble_preds_path == tmp_path / "model" / "ensemble" / "ensemble_preds"

@pytest.mark.unit
def test_eks_backend_rejects_nested_eks(tmp_path):
    with pytest.raises(ValueError, match="primitive backends"):
        EKSBackend(name="ensemble",
                   root_dir=tmp_path / "backend",
                   videos=[],
                   keypoints=[],
                   models={
                       "rng0": {"backend_type": "fake_primitive"},
                       "rng1": {"backend_type": "eks"},
                   })

@pytest.mark.unit
def test_eks_backend_rejects_mixed_backend_types(tmp_path):
    register_pose_backend("other_primitive", FakePrimitiveBackend)
    with pytest.raises(ValueError, match="same backend_type"):
        EKSBackend(name="ensemble",
                   root_dir=tmp_path / "backend",
                   videos=[],
                   keypoints=[],
                   models={
                       "rng0": {"backend_type": "fake_primitive"},
                       "rng1": {"backend_type": "other_primitive"},
                   })

@pytest.mark.unit
def test_eks_backend_rejects_unknown_train_model(tmp_path):
    with pytest.raises(ValueError, match="train_model"):
        EKSBackend(name="ensemble",
                   root_dir=tmp_path / "backend",
                   videos=[],
                   keypoints=[],
                   train_model="missing",
                   models={
                       "rng0": {"backend_type": "fake_primitive"},
                       "rng1": {"backend_type": "fake_primitive"},
                   })

@pytest.mark.unit
def test_eks_run_uses_regex_view_names(tmp_path, monkeypatch):
    backend = EKSBackend(name="ensemble",
                         root_dir=tmp_path / "backend",
                         videos=[],
                         keypoints=[KeypointConfig(label="nose")],
                         camera_names=["TL", "TR"],
                         view_names={"topleft": "TL", "topright": "TR"},
                         models={
                             "rng0": {"backend_type": "fake_primitive"},
                             "rng1": {"backend_type": "fake_primitive"},
                         })
    videos = {
        "topleft": tmp_path / "recording_TL.avi",
        "topright": tmp_path / "recording_TR.avi",
    }
    model_csvs = {
        "rng0": {path.resolve(): tmp_path / f"rng0-{path.stem}.csv"
                 for path in videos.values()},
        "rng1": {path.resolve(): tmp_path / f"rng1-{path.stem}.csv"
                 for path in videos.values()},
    }
    received = {}

    def fake_fit_eks_multicam(**kwargs):
        received.update(kwargs)
        return [pd.DataFrame(), pd.DataFrame()], None

    monkeypatch.setattr("cheese3d.backends.eks.fit_eks_multicam", fake_fit_eks_multicam)
    monkeypatch.setattr("cheese3d.backends.eks.dlc_df_to_h5", lambda *args: None)
    calibration = tmp_path / "calibration.toml"
    calibration.touch()
    backend._run_eks(model_csvs, videos, tmp_path / "output", calibration)

    assert set(received["input_source"]) == {"TL", "TR"}
    assert set(received["input_source"]) != set(videos)
