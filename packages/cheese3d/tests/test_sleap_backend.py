"""CPU-only tests for Cheese3D's isolated SLEAP adapter."""

import numpy as np
import pytest

sleap_io = pytest.importorskip("sleap_io")

from cheese3d.backends.sleap import (
    SLEAP_BACKBONES,
    _set_sleap_backbone,
    create_sleap_labels,
    create_sleap_training_config,
)


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
