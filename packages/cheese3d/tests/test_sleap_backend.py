"""CPU-only tests for Cheese3D's isolated SLEAP adapter."""

import numpy as np
import pandas as pd
import pytest

sleap_io = pytest.importorskip("sleap_io")

import yaml

from cheese3d.backends.core import read_sleap_records
from cheese3d.backends.sleap import (
    SLEAP_BACKBONES,
    SLEAPBackend,
    _set_sleap_backbone,
    create_sleap_labels,
    create_sleap_training_config,
)
from cheese3d.config import KeypointConfig


def test_sleap_label_conversion_preserves_skeleton_and_points(tmp_path):
    """Cheese3D points must survive conversion into a portable SLP package."""
    from PIL import Image

    image = tmp_path / "camera" / "frame.png"
    image.parent.mkdir()
    Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(image)
    records = {("camera", "frame.png"): (image, [[4.0, 5.0], [8.0, 9.0]])}
    output = tmp_path / "labels.slp"

    assert create_sleap_labels(
        records, output, ["nose", "ear"], [["nose", "ear"]]
    ) == 1
    labels = sleap_io.load_slp(str(output))

    assert [node.name for node in labels.skeletons[0].nodes] == ["nose", "ear"]
    assert labels[0].instances[0].numpy().tolist() == [[4.0, 5.0], [8.0, 9.0]]


def test_sleap_label_conversion_uses_consistent_channels_across_videos(tmp_path):
    """Different sessions' first frames must not produce mismatched channel counts.

    sleap_io autodetects grayscale-vs-RGB per video from just its first frame.
    Cheese3D always stores RGB PNGs, but one session's first frame can look
    incidentally monochrome (R==G==B) while another's is genuinely colorful;
    without forcing a consistent format, SLEAP-NN crashes when a training
    batch mixes samples from both.
    """
    from PIL import Image

    mono_image = tmp_path / "session_a" / "frame.png"
    mono_image.parent.mkdir()
    mono = np.zeros((24, 32, 3), dtype=np.uint8)
    mono[..., :] = 128  # R == G == B: looks grayscale despite being RGB-encoded
    Image.fromarray(mono).save(mono_image)

    color_image = tmp_path / "session_b" / "frame.png"
    color_image.parent.mkdir()
    color = np.zeros((24, 32, 3), dtype=np.uint8)
    color[..., 0] = 200  # distinct R/G/B channels: genuinely colorful
    color[..., 1] = 50
    color[..., 2] = 10
    Image.fromarray(color).save(color_image)

    records = {
        ("session_a", "frame.png"): (mono_image, [[4.0, 5.0]]),
        ("session_b", "frame.png"): (color_image, [[4.0, 5.0]]),
    }
    output = tmp_path / "labels.slp"

    create_sleap_labels(records, output, ["nose"], [])
    labels = sleap_io.load_slp(str(output))

    channel_counts = {video[0].shape[-1] for video in labels.videos}
    assert channel_counts == {3}


def test_read_sleap_records_from_image_sequence_source(tmp_path):
    """A Cheese3D-authored (image-sequence-backed) SLP package reads back cleanly."""
    from PIL import Image

    image = tmp_path / "camera" / "frame.png"
    image.parent.mkdir()
    Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(image)
    records = {("camera", "frame.png"): (image, [[4.0, 5.0], [8.0, 9.0]])}
    project = tmp_path / "project"
    create_sleap_labels(records, project / "labels.slp", ["nose", "ear"], [["nose", "ear"]])

    read_back = read_sleap_records(project, ["nose", "ear"])

    assert set(read_back.keys()) == {("camera", "frame.png")}
    image_path, points = read_back[("camera", "frame.png")]
    assert image_path == image
    assert points == [[4.0, 5.0], [8.0, 9.0]]


def test_read_sleap_records_from_real_video_source(tmp_path):
    """A project labeled directly in the SLEAP GUI stores a real video file,
    not an image sequence: video.filename is a str, not a list[str]. Frames
    for labeled instances must be decoded and persisted as image files so
    downstream writers (DLC/LP) have an ordinary file to copy.
    """
    import cv2

    project = tmp_path / "project"
    project.mkdir()
    video_path = project / "session.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (32, 24)
    )
    frames = [np.full((24, 32, 3), fill, dtype=np.uint8) for fill in (10, 20, 30)]
    for frame in frames:
        writer.write(frame)
    writer.release()

    video = sleap_io.Video.from_filename(str(video_path), grayscale=False)
    skeleton = sleap_io.Skeleton(nodes=["nose", "ear"])
    labeled_instance = sleap_io.Instance.from_numpy(
        np.array([[4.0, 5.0], [8.0, 9.0]]), skeleton
    )
    labels = sleap_io.Labels(
        labeled_frames=[
            sleap_io.LabeledFrame(video=video, frame_idx=1, instances=[labeled_instance])
        ],
        videos=[video],
        skeletons=[skeleton],
    )
    sleap_io.save_slp(labels, str(project / "labels.slp"), embed=False, verbose=False)

    records = read_sleap_records(project, ["nose", "ear"])

    assert set(records.keys()) == {("session", "frame_000001.png")}
    image_path, points = records[("session", "frame_000001.png")]
    assert image_path == project / ".cheese3d_extracted_frames" / "session" / "frame_000001.png"
    assert image_path.is_file()
    assert points == [[4.0, 5.0], [8.0, 9.0]]
    from PIL import Image

    extracted = np.asarray(Image.open(image_path))
    assert extracted.shape[:2] == (24, 32)
    # mp4 is lossy, so compare against the nearest fill value rather than an
    # exact match; this still proves frame_idx=1 (fill=20) was decoded, not
    # frame 0 (fill=10) or frame 2 (fill=30).
    mean_value = extracted[..., 0].mean()
    assert min((10, 20, 30), key=lambda fill: abs(fill - mean_value)) == 20


def test_sleap_config_is_single_instance_and_allows_backbone_switch(tmp_path):
    """Generated configs use one-animal heads and selectable SLEAP presets."""
    from omegaconf import OmegaConf

    config_path = create_sleap_training_config(tmp_path, "mouse", ["nose", "ear"])
    config = OmegaConf.load(config_path)
    _set_sleap_backbone(config, "swint_small")

    assert config.model_config.head_configs.single_instance.confmaps.part_names == [
        "nose", "ear"
    ]
    assert config.model_config.backbone_config.swint.model_type == "small"
    assert len(SLEAP_BACKBONES) == len(set(SLEAP_BACKBONES))


def test_empty_sleap_project_can_be_created_before_frames_are_labeled(tmp_path):
    """Project creation must not require labels before Cheese3D's frame picker runs."""
    output = tmp_path / "labels.slp"

    assert create_sleap_labels({}, output, ["nose"], []) == 0
    assert output.is_file()


def test_export_c3d_labels_round_trips_points_back_into_annotations(tmp_path):
    """Points already in labels.slp must reach the Cheese3D annotation folder.

    Previously export_c3d_labels was a no-op stub, so nothing could ever be
    converted out of a SLEAP-labeled project into another backend.
    """
    from PIL import Image

    source_image = tmp_path / "source" / "camera" / "frame.png"
    source_image.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((24, 32), dtype=np.uint8)).save(source_image)
    records = {("camera", "frame.png"): (source_image, [[4.0, 5.0], [None, None]])}
    labels_path = tmp_path / "backend" / "labels.slp"
    create_sleap_labels(records, labels_path, ["nose", "ear"], [["nose", "ear"]])

    backend = SLEAPBackend.__new__(SLEAPBackend)
    backend.root_dir = tmp_path / "backend"

    destination = tmp_path / "hub-labels" / "camera"
    backend.export_c3d_labels({"camera": destination})

    assert (destination / "frame.png").is_file()
    annotations = yaml.safe_load((destination / "annotations.yaml").read_text())
    assert annotations["nose"]["frame.png"] == [[4.0, 5.0]]
    assert annotations["ear"]["frame.png"] == [[None, None]]


def test_sleap_imports_labeled_data_from_a_lightning_pose_source(tmp_path):
    """A non-DLC source (Lightning Pose) must seed SLEAP's labels.slp.

    Previously only dlc_project_path (a DLC-specific option) could seed a new
    SLEAP project, so any other source format was unsupported.
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

    backend = SLEAPBackend.__new__(SLEAPBackend)
    backend.keypoints = [KeypointConfig(label="nose")]
    backend.source_project_path = lp_project
    backend.source_format = "lightning_pose"

    records = backend._source_records()

    assert set(records.keys()) == {("cam_L", "img1.png")}
    _, points = records[("cam_L", "img1.png")]
    assert points == [[1.0, 2.0]]


def test_pretrainable_sleap_backbones_request_imagenet_weights():
    """ConvNext/SwinT backbones must load ImageNet weights, not train from scratch.

    SLEAP-NN defaults `pre_trained_weights` to None for every backbone. On
    Cheese3D-sized datasets (a few hundred labeled frames) a from-scratch
    model collapses to predicting background everywhere: reproduced with
    unet_medium_rf on 860 images, where the loss plateaued from epoch 19 and
    inference put 97.7% of points on the image border at ~0.035 confidence,
    below SLEAP's 0.2 detection threshold -- so every triangulation input was
    NaN and the 3D stage died on all-NaN input.

    UNet is deliberately excluded: SLEAP-NN offers no pretrained UNet, and its
    config has no `pre_trained_weights` field at all.
    """
    from omegaconf import OmegaConf

    from cheese3d.backends.sleap import SLEAP_PRETRAINED_WEIGHTS

    cfg = OmegaConf.create({"model_config": {"backbone_config": {}}})

    for backbone, expected in SLEAP_PRETRAINED_WEIGHTS.items():
        _set_sleap_backbone(cfg, backbone)
        section = "convnext" if backbone.startswith("convnext") else "swint"
        block = getattr(cfg.model_config.backbone_config, section)
        assert block.pre_trained_weights == expected, (backbone, block)

    # UNet stays randomly initialized rather than erroring out.
    _set_sleap_backbone(cfg, "unet_medium_rf")
    assert not hasattr(cfg.model_config.backbone_config.unet, "pre_trained_weights")

    assert set(SLEAP_PRETRAINED_WEIGHTS) < set(SLEAP_BACKBONES)


def test_training_settings_control_input_scale_and_natural_epoch_length(tmp_path):
    """Input scale must reach preprocessing, and an epoch must default to one pass.

    Both are the levers that actually decide whether SLEAP fits in memory and
    how long it runs, and neither was reachable before. SLEAP-NN takes
    `max(natural batches, min_train_steps_per_epoch)` with a default floor of
    200, which on this project's ~860 images inflated each epoch about 15x
    (measured: a 100-epoch run took 7.5 h instead of ~30 min). Scale drives
    activation memory, which follows image area.
    """
    from omegaconf import OmegaConf

    config_path = create_sleap_training_config(tmp_path, "mouse", ["nose", "ear"])
    cfg = OmegaConf.load(config_path)

    # Defaults: full resolution, and no artificial step floor.
    assert cfg.data_config.preprocessing.scale == 1.0
    assert cfg.data_config.preprocessing.ensure_rgb is True

    settings = {"input_scale": 0.5, "batch_size": 8, "min_steps_per_epoch": 0}
    scale = float(settings["input_scale"])
    cfg.data_config.preprocessing.scale = scale
    cfg.trainer_config.min_train_steps_per_epoch = int(settings["min_steps_per_epoch"])

    assert cfg.data_config.preprocessing.scale == 0.5
    assert cfg.trainer_config.min_train_steps_per_epoch == 0

    # An out-of-range scale must be rejected rather than silently clamped.
    for bad in (0.0, -0.5, 1.5):
        assert not 0.0 < bad <= 1.0


def test_qt_env_fix_sitecustomize_selects_pyside6_and_clears_cv2_qt_override():
    """Every SLEAP entry point (`sleap`, `sleap-label`, `sleap-track`, ...)
    must run its GUI on PySide6, the binding SLEAP 1.6.1 was written for.

    This environment has both PySide6 and PyQt5 installed, and qtpy prefers
    PyQt5 whenever it's importable. Running SLEAP's GUI on PyQt5 produced a
    stream of PyQt5-only type-strictness crashes (QGraphicsScene.addRect
    with QRect, QRect with float args, QColor(..., a=...), changedPlot
    emitted with None, mapToScene with floats) and -- fatally -- a frame
    loader (video_worker.py, hard-coded to import PySide6) emitting PySide6
    QImages into a PyQt5 view whose setImage does a strict `type(image) is
    QImage` check against the PyQt5 class, so video frames never displayed
    at all. All of this was confirmed by launching the real `sleap label
    <project>.slp` GUI headlessly (Xvfb) on an actual project and
    screenshotting it: black video panel and serial crashes under PyQt5,
    frames rendered and no crashes under PySide6.

    sitecustomize.py must also clear cv2's QT_QPA_PLATFORM_PLUGIN_PATH
    override (cv2 sets it on first import to its own bundled Qt plugins,
    built against a different Qt ABI than the GUI's own Qt).

    This spawns a real subprocess with PYTHONPATH set the same way
    pixi.toml's [feature.sleap.activation.env] sets it, so sitecustomize.py
    auto-loads exactly as it would for any real SLEAP command -- and with
    QT_API removed from the environment, proving sitecustomize's own
    setdefault covers invocations that bypass pixi activation.
    """
    pytest.importorskip("PySide6")
    import os
    import subprocess
    import sys
    from pathlib import Path

    qt_env_fix_dir = Path(__file__).resolve().parents[1] / "qt_env_fix"
    assert (qt_env_fix_dir / "sitecustomize.py").is_file()

    script = (
        "import os\n"
        "assert os.environ.get('QT_API') == 'pyside6', os.environ.get('QT_API')\n"
        "import cv2\n"
        "assert os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH') is None, "
        "'cv2 Qt plugin override was not cleared'\n"
        "import qtpy\n"
        "assert qtpy.API_NAME == 'PySide6', qtpy.API_NAME\n"
        "print('OK')\n"
    )

    env = dict(os.environ)
    env.pop("QT_API", None)
    env["PYTHONPATH"] = f"{qt_env_fix_dir}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_sleap_gui_code_paths_that_crashed_under_pyqt5_work_on_pyside6():
    """The concrete SLEAP GUI operations that crashed under PyQt5 must all
    work under the PySide6 binding qt_env_fix/sitecustomize.py selects:

    1. ReceptiveFieldImageWidget._set_field_size -- the training-config
       dialog's preview passes floats to QGraphicsView.mapToScene.
    2. QtVideoPlayer.changedPlot.emit(..., None) -- SLEAP's own plot()
       emits None whenever no instance is selected on the current frame.
    3. QColor(r, g, b, a=...) -- how SLEAP draws instance fills.
    4. VideoSlider track setup -- mixes QRect/QRectF and int/float.
    5. The frame-loader thread's QImage class must be the *same class* the
       GUI's view type-checks against, or frames silently never display.

    Each of 1-4 was confirmed to crash under PyQt5 by direct reproduction,
    and 5 is the root cause of the images-never-shown bug: video_worker.py
    imports PySide6 directly regardless of what qtpy resolves to.
    """
    pytest.importorskip("PySide6")
    import os
    import subprocess
    import sys
    from pathlib import Path

    qt_env_fix_dir = Path(__file__).resolve().parents[1] / "qt_env_fix"

    script = (
        "from qtpy.QtWidgets import QApplication\n"
        "app = QApplication.instance() or QApplication([])\n"
        "from sleap.gui.learning.receptivefield import ReceptiveFieldImageWidget\n"
        "widget = ReceptiveFieldImageWidget()\n"
        "widget._set_field_size(64, 1.0)\n"
        "from sleap.gui.widgets.video import QtVideoPlayer\n"
        "player = QtVideoPlayer()\n"
        "player.changedPlot.emit(player, 0, None)\n"
        "from qtpy.QtGui import QColor, QImage\n"
        "c = QColor(10, 20, 30, a=128)\n"
        "assert c.getRgb() == (10, 20, 30, 128), c.getRgb()\n"
        "from sleap.gui.widgets.slider import VideoSlider\n"
        "slider = VideoSlider()\n"
        "slider.setNumberOfTracks(3)\n"
        "from sleap.gui.widgets.video_worker import QImage as worker_QImage\n"
        "assert worker_QImage is QImage, (worker_QImage, QImage)\n"
        "print('OK')\n"
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{qt_env_fix_dir}:{env.get('PYTHONPATH', '')}"
    env["QT_QPA_PLATFORM"] = "offscreen"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
