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
    thresholds = []

    monkeypatch.setattr(
        "cheese3d.generate_videos._read_pose_2d",
        lambda *_args, **_kwargs: (["nose"], None, None),
    )

    def fake_visualize(_scheme, _bodyparts, _points, _scores, _video, output, **_kwargs):
        Path(output).touch()
        outputs.append(Path(output))
        thresholds.append(_kwargs["probability_threshold"])

    monkeypatch.setattr("cheese3d.generate_videos.visualize_labels", fake_visualize)

    completed = generate_videos_2d(
        [], ["nose"], raw_dir, pose_dir, out_dir, probability_threshold=0.7
    )

    assert completed == 1
    assert outputs == [out_dir / "recording_TL.mp4"]
    assert thresholds == [0.7]


def test_changed_probability_threshold_regenerates_cached_video(monkeypatch, tmp_path):
    """A new GUI p cutoff must not silently reuse an overlay made with the old cutoff."""
    raw_dir, pose_dir, out_dir = (tmp_path / name for name in ("raw", "pose", "out"))
    raw_dir.mkdir()
    pose_dir.mkdir()
    video = raw_dir / "recording_TL.avi"
    pose = pose_dir / "recording_TL.h5"
    video.touch()
    pose.touch()
    rendered = []
    monkeypatch.setattr(
        "cheese3d.generate_videos._read_pose_2d",
        lambda *_args, **_kwargs: (["nose"], None, None),
    )
    monkeypatch.setattr("cheese3d.generate_videos.get_nframes", lambda _path: 100)

    def fake_visualize(_scheme, _bodyparts, _points, _scores, _video, output, **kwargs):
        """Record the cutoff carried into the camera renderer."""
        Path(output).touch()
        rendered.append(kwargs["probability_threshold"])

    monkeypatch.setattr("cheese3d.generate_videos.visualize_labels", fake_visualize)
    generate_videos_2d([], ["nose"], raw_dir, pose_dir, out_dir,
                       probability_threshold=0.2)
    generate_videos_2d([], ["nose"], raw_dir, pose_dir, out_dir,
                       probability_threshold=0.8)

    assert rendered == [0.2, 0.8]
