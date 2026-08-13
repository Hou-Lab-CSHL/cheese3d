import subprocess

import numpy as np
import pandas as pd
import pytest

from cheese3d.backends.lightning_pose import (LightningPoseBackend,
                                               is_lightning_pose_video,
                                               _make_vit_resize_square,
                                               _scale_scheduler_milestones,
                                               _vit_needs_unused_parameter_ddp,
                                               preprocess_lightning_pose_video,
                                               read_lp_preds,
                                               convert_dlc_labels_to_lightning_pose,
                                               create_lightning_pose_training_config)
from cheese3d.config import KeypointConfig


def test_track_regenerates_predictions_when_a_checkpoint_is_selected(monkeypatch, tmp_path):
    """Selecting a checkpoint must force fresh predictions, not reuse stale CSVs.

    Without this, re-tracking an already-tracked video after switching
    checkpoints silently reused whatever prediction CSV was already on disk
    from a prior (possibly different) checkpoint.
    """
    monkeypatch.setattr(
        "lightning_pose.utils.predictions.predict_video", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "lightning_pose.api.model.load_model_from_checkpoint",
        lambda **kwargs: object(),
    )

    class FakeModel:
        class cfg:
            class training:
                test_batch_size = 32

        def __init__(self, preds_dir):
            self._preds_dir = preds_dir
            self.model = None

        def video_preds_dir(self):
            return self._preds_dir

        def predict_on_video_file(self, video, **kwargs):
            calls.append(video)

    calls = []
    preds_dir = tmp_path / "video_preds"
    preds_dir.mkdir()
    video = tmp_path / "video.mp4"
    video.touch()
    # A stale prediction CSV from an earlier (different) checkpoint already exists.
    (preds_dir / "video.csv").write_text(
        "scorer,,\nbodyparts,nose,nose\ncoords,x,y\nimage,1.0,2.0\n"
    )

    backend = LightningPoseBackend.__new__(LightningPoseBackend)
    backend.root_dir = tmp_path
    backend.model = FakeModel(preds_dir)
    backend.scorer = "tester"
    backend._selected_checkpoint = tmp_path / "checkpoint.ckpt"

    monkeypatch.setattr(
        "cheese3d.backends.lightning_pose.preprocess_lightning_pose_video",
        lambda video, output_dir: video,
    )
    monkeypatch.setattr(
        "cheese3d.backends.lightning_pose.lp_csv_to_dlc_h5",
        lambda *a, **k: None,
    )

    backend.track({"view": video}, tmp_path / "output")

    assert calls == [video], "predict_on_video_file must run despite an existing stale CSV"


def test_lightning_pose_milestones_follow_gui_epoch_limit():
    """A shorter run must not retain scheduler epochs beyond max_epochs."""
    assert _scale_scheduler_milestones([150, 200, 250], 300, 125) == [62, 83, 104]


def test_vit_backbone_uses_square_resize_without_increasing_memory():
    """DINO/ViT inputs use the smaller side while CNN dimensions are untouched."""
    from omegaconf import OmegaConf

    vit_cfg = OmegaConf.create({"data": {"image_resize_dims": {
        "height": 512, "width": 640,
    }}})
    assert _make_vit_resize_square(vit_cfg, "vits_dinov2") == 512
    assert dict(vit_cfg.data.image_resize_dims) == {"height": 512, "width": 512}

    cnn_cfg = OmegaConf.create({"data": {"image_resize_dims": {
        "height": 512, "width": 640,
    }}})
    assert _make_vit_resize_square(cnn_cfg, "resnet50") is None
    assert dict(cnn_cfg.data.image_resize_dims) == {"height": 512, "width": 640}


def test_unused_parameter_ddp_is_limited_to_multigpu_vit():
    """CNNs and single-GPU ViTs retain Lightning's faster default strategy."""
    assert _vit_needs_unused_parameter_ddp("vits_dinov2", 2)
    assert not _vit_needs_unused_parameter_ddp("vits_dinov2", 1)
    assert not _vit_needs_unused_parameter_ddp("resnet50_animal_ap10k", 2)

@pytest.mark.unit
def test_read_lightning_pose_predictions_filters_metadata_columns(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    columns = pd.MultiIndex.from_tuples([
        ("set", "", ""),
        ("model", "nose", "x"),
        ("model", "nose", "y"),
        ("model", "nose", "likelihood"),
        ("model", "nose", "z"),
    ])
    df = pd.DataFrame([["test", 1.0, 2.0, 0.9, 3.0]], columns=columns)
    df.to_csv(csv_path)
    parsed = read_lp_preds(csv_path, scorer="lp")
    assert list(parsed.columns) == [
        ("lp", "nose", "x"),
        ("lp", "nose", "y"),
        ("lp", "nose", "likelihood"),
    ]
    assert parsed.iloc[0].tolist() == [1.0, 2.0, 0.9]

@pytest.mark.unit
def test_lightning_pose_video_requires_mp4_h264_yuv420p(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    monkeypatch.setattr(
        "cheese3d.backends.lightning_pose.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0,
            '{"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"}, '
            '"streams": [{"codec_name": "h264", "pix_fmt": "yuv420p"}]}'
        )
    )
    assert is_lightning_pose_video(video)
    assert not is_lightning_pose_video(tmp_path / "video.avi")

@pytest.mark.unit
def test_preprocess_lightning_pose_video_uses_app_ffmpeg_settings(monkeypatch, tmp_path):
    video = tmp_path / "video.avi"
    video.touch()
    commands = []
    monkeypatch.setattr("cheese3d.backends.lightning_pose.is_lightning_pose_video",
                        lambda path: False)
    monkeypatch.setattr("cheese3d.backends.lightning_pose.subprocess.run",
                        lambda command, **kwargs: commands.append(command))
    output_path = preprocess_lightning_pose_video(video, tmp_path / "preprocessed-videos")
    assert output_path == tmp_path / "preprocessed-videos" / "video.mp4"
    assert commands == [[
        "ffmpeg", "-i", str(video), "-loglevel", "info", "-stats",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "1",
        "-preset", "medium", "-crf", "23", "-vf", "setsar=1", "-an",
        "-y", str(output_path),
    ]]


@pytest.mark.unit
def test_dlc_labels_convert_to_portable_lightning_pose_data(tmp_path):
    """DLC tables with different schemas merge without losing images or rows."""
    dlc_project = tmp_path / "dlc"
    for folder_name, bodyparts in (("cam_L", ["nose", "ear"]),
                                   ("cam_R", ["nose"])):
        label_dir = dlc_project / "labeled-data" / folder_name
        label_dir.mkdir(parents=True)
        image_name = f"{folder_name}.png"
        (label_dir / image_name).write_bytes(b"image")
        columns = pd.MultiIndex.from_tuples([
            ("tester", bodypart, coord)
            for bodypart in bodyparts for coord in ("x", "y")
        ])
        index = pd.MultiIndex.from_tuples([
            ("labeled-data", folder_name, image_name)
        ])
        pd.DataFrame([[float(i) for i in range(len(columns))]],
                     index=index, columns=columns).to_hdf(
                         label_dir / "CollectedData_tester.h5", key="df"
                     )

    output = tmp_path / "lp"
    summary = convert_dlc_labels_to_lightning_pose(
        dlc_project, output, keypoint_names=["nose", "ear"]
    )
    config_path = create_lightning_pose_training_config(
        output, summary, model_name="converted"
    )
    converted = pd.read_csv(summary["csv_file"], header=[0, 1, 2], index_col=0)
    raw_csv_lines = summary["csv_file"].read_text().splitlines()

    assert summary["num_frames"] == 2
    assert summary["num_keypoints"] == 2
    assert list(converted.columns.get_level_values(1).unique()) == ["nose", "ear"]
    assert pd.isna(converted.loc["labeled-data/cam_R/cam_R.png",
                                 ("lightning_pose", "ear", "x")])
    # The first training sample must immediately follow LP's three header rows.
    assert raw_csv_lines[3].startswith("labeled-data/")
    assert "image" not in converted.index
    assert all((summary["data_dir"] / path).is_file() for path in converted.index)
    assert config_path.is_file()


def test_lp_imports_labeled_data_from_a_sleap_source(tmp_path):
    """A non-DLC source (SLEAP) must seed a real LP CollectedData.csv.

    Previously only dlc_project_path (a DLC-specific option) could seed a new
    LP project, so any other source format was unsupported.
    """
    sleap_io = pytest.importorskip("sleap_io")
    from PIL import Image
    from cheese3d.backends.sleap import create_sleap_labels

    image = tmp_path / "camera" / "frame.png"
    image.parent.mkdir()
    Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(image)
    records = {("camera", "frame.png"): (image, [[4.0, 5.0]])}
    sleap_project = tmp_path / "sleap-source"
    create_sleap_labels(records, sleap_project / "labels.slp", ["nose"], [])

    backend = LightningPoseBackend.__new__(LightningPoseBackend)
    backend.root_dir = tmp_path / "lp"
    backend.name = "model"
    backend.keypoints = [KeypointConfig(label="nose")]
    backend.source_project_path = sleap_project
    backend.source_format = "sleap"
    backend.project_path.mkdir(parents=True)

    summary = backend._import_source_project()

    assert summary["num_frames"] == 1
    csv_path = backend.project_path / "data" / "CollectedData.csv"
    assert csv_path.is_file()
    table = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    assert table.iloc[0][("lightning_pose", "nose", "x")] == 4.0
    destination = backend.project_path / "data" / "labeled-data" / "camera" / "frame.png"
    assert destination.is_file()


def test_vit_backbones_fall_back_to_a_single_gpu(capsys):
    """ViT/DINO training works across GPUs but its prediction pass hangs.

    Lightning Pose builds a second Trainer for the post-training
    "Predicting train/val/test images..." pass while the training process
    group is still up, and rank 0 then blocks forever in a collective. The
    signature is one GPU pinned at 100% holding ~1.4 GiB with zero prediction
    batches after 15 minutes -- an NCCL spin-wait, not compute. Confirmed by
    direct comparison: the same model on one GPU reached 93 prediction batches
    in 25 seconds, so falling back makes these backbones usable rather than
    leaving them to hang.
    """
    from cheese3d.backends.lightning_pose import _supported_training_gpus

    assert _supported_training_gpus("vits_dino", ["0", "1"]) == ["0"]
    assert _supported_training_gpus("vitb_dinov2", ["1", "0"]) == ["1"]
    assert "hangs in its post-training prediction" in capsys.readouterr().out

    # Already single-GPU (or CPU) runs pass through untouched.
    assert _supported_training_gpus("vits_dino", ["1"]) == ["1"]
    assert _supported_training_gpus("vits_dino", []) == []

    # CNN backbones keep every GPU they were given.
    assert _supported_training_gpus("resnet50", ["0", "1"]) == ["0", "1"]
    assert _supported_training_gpus("efficientnet_b0", ["0", "1"]) == ["0", "1"]
