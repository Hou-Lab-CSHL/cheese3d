import pandas as pd
import yaml
from pathlib import Path

from cheese3d.backends.dlc import (
    DLC3_PYTORCH_MODELS,
    DLCBackend,
    _enforce_dlc3_project_config,
    _include_compatible_labeled_data,
    _paf_graph_from_skeleton,
    _shuffle_from_created_splits,
)
from cheese3d.config import KeypointConfig


def test_dlc3_config_removes_legacy_pose_fields_and_uses_pytorch(tmp_path):
    """Imported DLC2 pose options must not enter DLC3 project validation."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "Task: demo\nengine: tensorflow\niteration: 4\nweight_decay: 0.0001\n"
    )

    removed = _enforce_dlc3_project_config(
        config_path, {"Task", "engine", "iteration", "config_version"}
    )
    migrated = yaml.safe_load(config_path.read_text())

    assert removed == ["weight_decay"]
    assert migrated == {
        "Task": "demo", "engine": "pytorch", "iteration": 4,
        "config_version": 0,
    }


def test_dlc3_tracking_passes_selected_snapshot_index(monkeypatch, tmp_path):
    """DLC3 PyTorch tracking requires `snapshot_index`, not DLC2's spelling."""
    calls = {}
    monkeypatch.setattr(
        "deeplabcut.analyze_videos",
        lambda **kwargs: calls.update(kwargs),
    )
    backend = DLCBackend.__new__(DLCBackend)
    backend.root_dir = tmp_path
    backend.name = "model"
    backend.experimenter = "tester"
    backend.date = "2026-01-01"
    backend._selected_snapshot_index = 1
    backend._selected_shuffle = 6
    video = tmp_path / "video.mp4"
    video.touch()

    assert backend.track({"view": video}, tmp_path / "pose") is True
    assert calls["snapshot_index"] == 1
    assert calls["shuffle"] == 6
    assert "snapshotindex" not in calls
    assert calls["device"] == "cuda:0"
    assert calls["batch_size"] == 8


def test_dlc3_training_uses_selected_network_architecture(monkeypatch, tmp_path):
    """The Training-tab model choice must replace the old fixed ResNet-50."""
    calls = {}
    monkeypatch.setattr(
        "deeplabcut.pose_estimation_pytorch.available_models",
        lambda: ["resnet_50", "hrnet_w32"],
    )
    monkeypatch.setattr(
        "deeplabcut.create_training_dataset",
        lambda **kwargs: calls.setdefault("dataset", kwargs),
    )
    monkeypatch.setattr(
        "deeplabcut.train_network",
        lambda **kwargs: calls.setdefault("train", kwargs),
    )
    monkeypatch.setattr(
        "deeplabcut.evaluate_network",
        lambda **kwargs: calls.setdefault("evaluate", kwargs),
    )
    monkeypatch.setattr("torch.cuda.set_device", lambda device: calls.setdefault("primary_gpu", device))
    backend = DLCBackend.__new__(DLCBackend)
    backend.root_dir = tmp_path
    backend.name = "model"
    backend.experimenter = "tester"
    backend.date = "2026-01-01"
    # train() now re-syncs via overwrite_config() before launching DLC, so the
    # test double needs the same attributes __init__ would have set.
    backend.videos = []
    backend.crops = []
    backend.keypoints = []
    backend.skeleton = []
    backend.frames_per_video = 5
    backend.canonical_config_path = None
    backend.project_path.mkdir(parents=True)
    backend.config_path.write_text("Task: demo\n")

    backend.train("0", iterate_dataset=False, training_settings={
        "network_architecture": "hrnet_w32", "epochs": 1, "batch_size": 1,
        "save_every_n_epochs": 1, "train_fraction_percent": 80,
        "max_snapshots_to_keep": 3,
    })

    assert calls["dataset"]["net_type"] == "hrnet_w32"
    assert calls["dataset"]["Shuffles"] == [1]
    assert calls["train"]["shuffle"] == 1
    assert calls["evaluate"]["shuffles"] == [1]
    assert calls["train"]["max_snapshots_to_keep"] == 3
    assert calls["train"]["device"] == "cuda"
    assert calls["primary_gpu"] == 0
    assert yaml.safe_load(backend.config_path.read_text())["TrainingFraction"] == [0.8]


def test_dlc_imports_labeled_data_from_a_lightning_pose_source(tmp_path):
    """A non-DLC source (Lightning Pose) must seed real DLC labeled-data.

    This must not require deeplabcut: reading a foreign LP source and writing
    DLC's CollectedData layout are both pure-pandas operations, independent of
    which Pixi environment (and thus which pose package) is installed.
    """
    lp_project = tmp_path / "lp-source"
    data_dir = lp_project / "data"
    label_dir = data_dir / "labeled-data" / "cam_L"
    label_dir.mkdir(parents=True)
    (label_dir / "img1.png").write_bytes(b"image")
    columns = pd.MultiIndex.from_tuples([
        ("lightning_pose", bodypart, coord)
        for bodypart in ["nose"] for coord in ("x", "y")
    ], names=["scorer", "bodyparts", "coords"])
    pd.DataFrame([[1.0, 2.0]], index=["labeled-data/cam_L/img1.png"],
                columns=columns).to_csv(data_dir / "CollectedData.csv")

    backend = DLCBackend.__new__(DLCBackend)
    backend.root_dir = tmp_path
    backend.name = "model"
    backend.experimenter = "tester"
    backend.date = "2026-01-01"
    backend.keypoints = [KeypointConfig(label="nose")]
    backend.source_project_path = lp_project
    backend.source_format = "lightning_pose"
    backend.project_path.mkdir(parents=True)

    summary = backend._import_source_project()

    assert summary == {"folders": 1, "images": 1, "records": 1}
    written = backend.project_path / "labeled-data" / "cam_L"
    assert (written / "img1.png").is_file()
    table = pd.read_hdf(written / "CollectedData_tester.h5")
    assert table.iloc[0][("tester", "nose", "x")] == 1.0
    assert table.iloc[0][("tester", "nose", "y")] == 2.0


def test_single_animal_model_selector_includes_dlcrnet_with_cheese3d_pafs():
    """DLCRNet remains selectable when Cheese3D can supply its skeleton as PAFs."""
    assert "dlcrnet_stride16_ms5" in DLC3_PYTORCH_MODELS
    assert _paf_graph_from_skeleton(
        ["nose", "eye", "ear"], [["nose", "eye"], ["eye", "ear"]]
    ) == [[0, 1], [1, 2]]


def test_created_training_split_selects_new_shuffle():
    """A changed train fraction may create shuffle 2+, which training must use."""
    assert _shuffle_from_created_splits([(0.7, 2, ([0], [1]))]) == 2


def test_all_compatible_label_folders_enter_dlc_dataset_selection(tmp_path):
    """Imported labels without source videos must still enter DLC's merger."""
    project = tmp_path / "project"
    (project / "videos").mkdir(parents=True)
    labels_root = project / "labeled-data"
    columns = __import__("pandas").MultiIndex.from_product(
        [["tester"], ["nose", "tail"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    for folder_name in ("camera", "imported"):
        folder = labels_root / folder_name
        folder.mkdir(parents=True)
        (folder / "img001.png").touch()
        frame = __import__("pandas").DataFrame(
            [[1.0, 2.0, 3.0, 4.0]], columns=columns,
            index=__import__("pandas").MultiIndex.from_tuples(
                [("labeled-data", folder_name, "img001.png")]
            ),
        )
        frame.to_hdf(folder / "CollectedData_tester.h5", key="df")
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "project_path": str(project), "scorer": "tester",
        "bodyparts": ["nose", "tail"],
        "video_sets": {str(project / "videos" / "camera.mp4"): {"crop": "0, 1, 0, 1"}},
    }))

    summary = _include_compatible_labeled_data(config_path)
    updated = yaml.safe_load(config_path.read_text())

    assert summary == {"folders": 2, "images": 2, "skipped": []}
    assert {Path(path).stem for path in updated["video_sets"]} == {"camera", "imported"}


def test_compatible_bodyparts_can_use_a_different_column_order(tmp_path):
    """DLC reindexes bodyparts, so HDF ordering must not exclude valid labels."""
    project = tmp_path / "project"
    folder = project / "labeled-data" / "imported"
    folder.mkdir(parents=True)
    (folder / "img.png").touch()
    pandas = __import__("pandas")
    frame = pandas.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        columns=pandas.MultiIndex.from_product(
            [["tester"], ["tail", "nose"], ["x", "y"]],
            names=["scorer", "bodyparts", "coords"],
        ),
        index=pandas.MultiIndex.from_tuples(
            [("labeled-data", "imported", "img.png")]
        ),
    )
    frame.to_hdf(folder / "CollectedData_tester.h5", key="df")
    config_path = project / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "project_path": str(project), "scorer": "tester",
        "bodyparts": ["nose", "tail"], "video_sets": {},
    }))

    assert _include_compatible_labeled_data(config_path)["images"] == 1


def test_dlcrnet_falls_back_to_a_single_gpu(capsys):
    """DLCRNet cannot train under DLC's multi-GPU DataParallel wrapper.

    Its multi-scale branches are not replicated correctly, so training dies
    almost immediately with "Expected all tensors to be on the same device,
    but got weight is on cuda:0 ... and input is on cuda:1". Confirmed by
    direct comparison: both dlcrnet variants failed within ~19 s on two GPUs
    and trained a full epoch on one. Falling back is much better than failing,
    since the alternative is that the architecture is simply unusable.
    """
    from cheese3d.backends.dlc import _supported_training_gpus

    assert _supported_training_gpus("dlcrnet_stride16_ms5", [0, 1]) == [0]
    assert _supported_training_gpus("dlcrnet_stride32_ms5", [1, 0]) == [1]
    assert "does not support multi-GPU" in capsys.readouterr().out

    # A single GPU, or CPU, is already valid and must pass through untouched.
    assert _supported_training_gpus("dlcrnet_stride16_ms5", [1]) == [1]
    assert _supported_training_gpus("dlcrnet_stride16_ms5", []) == []

    # Every other architecture keeps both GPUs.
    assert _supported_training_gpus("resnet_50", [0, 1]) == [0, 1]
    assert _supported_training_gpus("hrnet_w32", [0, 1]) == [0, 1]


def test_batchnorm_architectures_default_to_a_lower_learning_rate(capsys):
    """DLC's BatchNorm backbones cannot train at ResNet's default rate.

    DLC3 builds ResNet with GroupNorm (``resnet50_gn``), which ignores batch
    statistics, but gives the other architectures ordinary BatchNorm with
    frozen pretrained running statistics. A rate that moves activations away
    from the distribution those statistics encode collapses the model to
    predicting background everywhere. Measured on hrnet_w32, everything else
    identical: 5e-4 held the training loss at 0.0149 for 30+ epochs with
    validation RMSE 419 px and mAP 0.00, while 1e-4 reached RMSE 23.04 px and
    mAP 87.58 in ten epochs.
    """
    from cheese3d.backends.dlc import default_learning_rate, resolve_learning_rate

    assert default_learning_rate("resnet_50") == 5e-4
    assert default_learning_rate("resnet_101") == 5e-4
    for architecture in ("hrnet_w32", "hrnet_w48", "dekr_w18", "cspnext_x",
                         "dlcrnet_stride16_ms5"):
        assert default_learning_rate(architecture) == 1e-4, architecture

    # An unset rate takes the architecture's own default.
    assert resolve_learning_rate("hrnet_w32", {}) == 1e-4
    assert resolve_learning_rate("resnet_50", {}) == 5e-4

    # An explicit rate is honoured, but an unsafe one warns rather than
    # silently reproducing the flat-loss failure.
    assert resolve_learning_rate("hrnet_w32", {"learning_rate": 5e-4}) == 5e-4
    assert "unstable above" in capsys.readouterr().out

    # Safe explicit rates pass without noise.
    assert resolve_learning_rate("hrnet_w32", {"learning_rate": 1e-5}) == 1e-5
    assert "unstable" not in capsys.readouterr().out


def test_gui_learning_rate_matches_the_backend_recommendation():
    """The GUI must not offer a rate the backend considers unsafe.

    interactive.py duplicates the rule rather than importing the DLC backend,
    because it is also loaded in the Lightning Pose and SLEAP environments
    where DeepLabCut is not installed. Check the two agree.
    """
    from cheese3d.backends.dlc import DLC3_PYTORCH_MODELS, default_learning_rate
    from cheese3d.interactive import _dlc_default_learning_rate

    for architecture in DLC3_PYTORCH_MODELS:
        assert _dlc_default_learning_rate(architecture) == default_learning_rate(
            architecture
        ), architecture
