"""Tests for the backend-aware settings schema and the config editor."""

import tempfile
from pathlib import Path

import pytest
import yaml

from cheese3d.project import Ch3DProject
from cheese3d.settings import (
    TRAINING_SETTINGS,
    backend_settings,
    describe_training_settings,
    validate_training_settings,
)


def test_schema_is_importable_from_every_backend_environment():
    """The schema must describe all three backends without importing any.

    DeepLabCut, Lightning Pose and SLEAP deliberately cannot share a Python
    process, so a schema that imported them could only ever be read from one
    environment -- and the GUI, which runs in whichever environment the user
    opened, needs all three.
    """
    assert set(TRAINING_SETTINGS) == {"dlc", "lightning_pose", "sleap"}
    for backend in TRAINING_SETTINGS:
        assert len(backend_settings(backend)) > 5
        assert "epochs" in backend_settings(backend)

    with pytest.raises(ValueError, match="Unknown backend"):
        backend_settings("deeplabcut")


def test_misspelled_setting_is_rejected_rather_than_silently_defaulted():
    """Backends read settings with ``.get(name, default)``.

    That means a typo does not fail -- it silently trains with the default,
    which looks like a setting that "did not take effect" and is very hard to
    notice afterwards.
    """
    with pytest.raises(ValueError, match="did you mean 'epochs'"):
        validate_training_settings("dlc", {"epoch": 5})


def test_setting_belonging_to_another_backend_names_that_backend():
    """The backends' settings differ, and mixing them up is an easy mistake."""
    with pytest.raises(ValueError, match="belongs to sleap"):
        validate_training_settings("dlc", {"input_scale": 0.5})


@pytest.mark.parametrize("backend,settings", [
    ("sleap", {"input_scale": 2.0}),
    ("sleap", {"backbone": "not_a_real_backbone"}),
    ("dlc", {"train_fraction_percent": 150}),
    ("dlc", {"epochs": 0}),
    ("lightning_pose", {"val_prob": 1.5}),
])
def test_out_of_range_values_are_rejected(backend, settings):
    with pytest.raises(ValueError):
        validate_training_settings(backend, settings)


def test_relationships_between_settings_are_checked():
    """Some invalid combinations are only visible across two settings."""
    with pytest.raises(ValueError, match="must not exceed"):
        validate_training_settings("sleap", {"scale_min": 1.5, "scale_max": 0.9})

    with pytest.raises(ValueError, match="at most 1"):
        validate_training_settings(
            "lightning_pose", {"train_prob": 0.9, "val_prob": 0.5}
        )

    # A checkpoint interval longer than the run saves nothing, and the failure
    # only surfaces later when tracking finds no snapshot to load.
    with pytest.raises(ValueError, match="nothing"):
        validate_training_settings(
            "dlc", {"epochs": 5, "save_every_n_epochs": 20}
        )


def test_valid_settings_pass_and_are_coerced():
    assert validate_training_settings("sleap", {"epochs": "5"}) == {"epochs": 5}

    filled = validate_training_settings("dlc", {"epochs": 5}, fill_defaults=True)
    assert filled["epochs"] == 5
    assert filled["network_architecture"] == "resnet_50"

    # Lightning Pose's imgaug is a structured dict, not a scalar, so it passes
    # through without per-key validation.
    augmentation = {"MotionBlur": {"p": 0.0}}
    checked = validate_training_settings("lightning_pose", {"imgaug": augmentation})
    assert checked["imgaug"] == augmentation


def test_describe_lists_defaults_and_bounds():
    text = describe_training_settings("sleap")
    assert "input_scale" in text and "1.0" in text
    assert "convnext_tiny" in text


def test_sleap_defaults_match_sleap_nn():
    """Cheese3D must not silently differ from SLEAP's own defaults.

    A backend that quietly substitutes its own values makes results
    non-comparable to stock SLEAP and hides what is actually being run, so
    every default here is checked against the installed sleap-nn config
    classes rather than being asserted as a literal.
    """
    pytest.importorskip("sleap_nn")
    from sleap_nn.config.data_config import PreprocessingConfig
    from sleap_nn.config.model_config import SingleInstanceConfMapsConfig
    from sleap_nn.config.trainer_config import TrainerConfig

    schema = TRAINING_SETTINGS["sleap"]
    trainer, head = TrainerConfig(), SingleInstanceConfMapsConfig()
    preprocessing = PreprocessingConfig()

    assert schema["output_stride"].default == head.output_stride
    assert schema["input_scale"].default == preprocessing.scale
    assert schema["batch_size"].default == trainer.train_data_loader.batch_size
    assert schema["val_batch_size"].default == trainer.val_data_loader.batch_size
    assert schema["save_top_k"].default == trainer.model_ckpt.save_top_k
    assert schema["min_steps_per_epoch"].default == trainer.min_train_steps_per_epoch
    assert schema["learning_rate"].default == trainer.optimizer.lr
    assert schema["epochs"].default == trainer.max_epochs


def test_edit_config_updates_only_what_changed_and_validates_paths():
    """Hand-editing config.yaml misplaces keys; this reports what it changed."""
    directory = Path(tempfile.mkdtemp())
    (directory / "config.yaml").write_text(yaml.safe_dump({
        "name": "project", "fps": 100,
        "triangulation": {"score_threshold": 0.9, "filter2d": False},
    }))

    changed = Ch3DProject.edit_config(
        directory, fps=120, **{"triangulation.score_threshold": 0.6}
    )
    assert changed == {
        "fps": (100, 120), "triangulation.score_threshold": (0.9, 0.6),
    }

    written = yaml.safe_load((directory / "config.yaml").read_text())
    assert written["fps"] == 120
    assert written["triangulation"]["score_threshold"] == 0.6
    assert written["triangulation"]["filter2d"] is False  # untouched
    assert written["name"] == "project"

    # Setting a value to what it already is reports no change.
    assert Ch3DProject.edit_config(directory, fps=120) == {}

    with pytest.raises(ValueError, match="not in this configuration"):
        Ch3DProject.edit_config(directory, nonexistent=1)
    with pytest.raises(ValueError, match="not in this configuration"):
        Ch3DProject.edit_config(directory, **{"triangulation.nope": 1})
    with pytest.raises(ValueError, match="is a section, not a value"):
        Ch3DProject.edit_config(directory, triangulation=1)
    with pytest.raises(ValueError, match="No config.yaml"):
        Ch3DProject.edit_config(directory / "missing", fps=1)
