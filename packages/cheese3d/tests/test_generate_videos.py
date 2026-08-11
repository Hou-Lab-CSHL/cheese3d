from pathlib import Path

from cheese3d.generate_videos import _find_matching_video, generate_videos_2d


def test_find_matching_video_ignores_dlc_scorer_suffix(tmp_path):
    video = tmp_path / "recording_TL.avi"
    video.touch()

    match = _find_matching_video(
        "recording_TLDLC_resnet50_modelshuffle1_100", tmp_path
    )

    assert match == str(video)


def test_find_matching_video_accepts_exact_lp_stem(tmp_path):
    video = tmp_path / "recording_TL.mp4"
    video.touch()

    assert _find_matching_video("recording_TL", tmp_path) == str(video)


def test_generated_video_uses_raw_video_stem(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    pose_dir = tmp_path / "pose"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    pose_dir.mkdir()
    video = raw_dir / "recording_TL.avi"
    pose = pose_dir / "recording_TLDLC_model.h5"
    video.touch()
    pose.touch()
    outputs = []

    monkeypatch.setattr(
        "cheese3d.generate_videos._read_pose_2d",
        lambda *_args, **_kwargs: (["nose"], None, None),
    )

    def fake_visualize(_scheme, _bodyparts, _points, _scores, _video, output, **_kwargs):
        Path(output).touch()
        outputs.append(Path(output))

    monkeypatch.setattr("cheese3d.generate_videos.visualize_labels", fake_visualize)

    completed = generate_videos_2d([], ["nose"], raw_dir, pose_dir, out_dir)

    assert completed == 1
    assert outputs == [out_dir / "recording_TL.mp4"]
