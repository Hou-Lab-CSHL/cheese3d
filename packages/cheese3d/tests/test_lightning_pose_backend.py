import subprocess

import pandas as pd
import pytest

from cheese3d.backends.lightning_pose import (is_lightning_pose_video,
                                               preprocess_lightning_pose_video,
                                               read_lp_preds)

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
