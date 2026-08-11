import pytest
import toml
from pathlib import Path

from cheese3d.backends.core import get_pose_backend_class, register_pose_backend
from cheese3d.backends.dlc import DLCBackend
from cheese3d.backends.eks import EKSBackend
from cheese3d.backends.lightning_pose import LightningPoseBackend
from cheese3d.registry import RegistryError, load_entry_points
from cheese3d.synchronize.aligners import (BaseAligner,
                                           CrossCorrelationAligner,
                                           get_aligner_class,
                                           register_aligner)
from cheese3d.synchronize.core import SyncConfig
from cheese3d.synchronize.readers import (SyncSignalReader,
                                          get_ephys_reader,
                                          get_sync_reader_class,
                                          register_sync_reader)

class CustomAligner(BaseAligner):
    def align(self, ref_signal, target_signal, align_params=None):
        return align_params, None

class CustomSyncReader(SyncSignalReader):
    extra: str

    def __init__(self, source: Path, sample_rate: int, extra: str, **kwargs):
        super().__init__(source=source, sample_rate=sample_rate, **kwargs)
        self.extra = extra

    def load_signal(self):
        return []

    def root_path(self):
        return self.source

class CustomBackend:
    pass

class EntryPoint:
    def __init__(self, name, value=None, error=None):
        self.name = name
        self.value = value
        self.error = error

    def load(self):
        if self.error is not None:
            raise self.error

        return self.value

@pytest.mark.unit
def test_builtin_plugins_are_registered():
    assert get_aligner_class("crosscorr") is CrossCorrelationAligner
    assert get_pose_backend_class("dlc") is DLCBackend
    assert get_pose_backend_class("eks") is EKSBackend
    assert get_pose_backend_class("lightning_pose") is LightningPoseBackend


@pytest.mark.unit
def test_backend_extras_are_mutually_exclusive():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    config = toml.load(pyproject)
    extras = config["project"]["optional-dependencies"]
    dependencies = config["project"]["dependencies"]
    assert "dlc" in extras
    assert "lightning-pose" in extras
    assert "ensemble-kalman-smoother >= 0.1.0" in dependencies
    assert "ensemble-kalman-smoother >= 0.1.0" not in extras["dlc"]
    assert "ensemble-kalman-smoother >= 0.1.0" not in extras["lightning-pose"]
    assert config["tool"]["uv"]["conflicts"] == [
        [{"extra": "dlc"}, {"extra": "lightning-pose"}]
    ]

@pytest.mark.unit
def test_sync_config_builds_registered_aligner():
    register_aligner("custom", CustomAligner)
    cfg = SyncConfig(pipeline=[{"type": "custom", "debug": False}])
    pipeline = cfg.build_pipeline(ref_sample_rate=100, target_sample_rate=200)
    assert isinstance(pipeline[0], CustomAligner)
    assert pipeline[0].ref_sample_rate == 100
    assert pipeline[0].target_sample_rate == 200
    assert not pipeline[0].debug

@pytest.mark.unit
def test_ephys_reader_uses_registered_reader(tmp_path):
    register_sync_reader("custom", CustomSyncReader)
    reader = get_ephys_reader(tmp_path / "source.dat", {
        "type": "custom",
        "sample_rate": 30000,
        "sync_threshold": 0.4,
        "extra": "value"
    })
    assert isinstance(reader, CustomSyncReader)
    assert reader.sample_rate == 30000
    assert reader.threshold == 0.4
    assert reader.extra == "value"

@pytest.mark.unit
def test_pose_backend_registration():
    register_pose_backend("custom", CustomBackend)
    assert get_pose_backend_class("custom") is CustomBackend

@pytest.mark.unit
def test_entry_point_discovery(monkeypatch):
    from cheese3d.synchronize import readers

    def fake_load_entry_points(group, registry):
        assert group == "cheese3d.sync_readers"
        registry["entrypoint"] = CustomSyncReader

    monkeypatch.setattr(readers, "load_entry_points", fake_load_entry_points)
    readers.SYNC_READER_REGISTRY.pop("entrypoint", None)

    assert get_sync_reader_class("entrypoint") is CustomSyncReader

@pytest.mark.unit
def test_entry_point_conflict_reports_registered_item(monkeypatch):
    monkeypatch.setattr("cheese3d.registry.entry_points",
                        lambda group: [EntryPoint("existing", CustomAligner)])
    with pytest.raises(RegistryError, match="conflicts with registered item"):
        load_entry_points("cheese3d.aligners", {"existing": CustomBackend})

@pytest.mark.unit
def test_entry_point_load_failure_includes_plugin_context(monkeypatch):
    monkeypatch.setattr("cheese3d.registry.entry_points",
                        lambda group: [EntryPoint("broken", error=ImportError("missing dep"))])
    with pytest.raises(RegistryError, match="Failed to load plugin entry point 'broken'"):
        load_entry_points("cheese3d.aligners", {})
