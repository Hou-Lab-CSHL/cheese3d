from pathlib import Path

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
