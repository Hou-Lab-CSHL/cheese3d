import subprocess

import pandas as pd
import pytest

from cheese3d.backends.lightning_pose import (is_lightning_pose_video,
                                               _make_vit_resize_square,
                                               _scale_scheduler_milestones,
                                               _vit_needs_unused_parameter_ddp,
                                               preprocess_lightning_pose_video,
                                               read_lp_preds,
                                               convert_dlc_labels_to_lightning_pose,
                                               create_lightning_pose_training_config)


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
