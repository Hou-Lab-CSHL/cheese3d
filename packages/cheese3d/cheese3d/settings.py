"""Declarative schema for the settings each pose backend accepts.

``--training-settings`` and the GUI both hand the backends a plain dictionary,
and every backend reads it with ``settings.get(name, default)``. That is
forgiving in the worst way: a misspelled key such as ``"epoch"`` is silently
ignored and training quietly runs with the default instead, which looks like a
setting that "did not take effect" rather than an error. The same applies to a
value a backend cannot use -- a SLEAP ``input_scale`` of 2.0, say.

This module describes the settings as data so they can be validated,
discovered, and edited without importing any backend. That matters because the
three backends deliberately cannot share a Python process: the Lightning Pose
environment has no DeepLabCut installed and vice versa, so a schema that
imported them could never be loaded from one place.
"""

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Setting:
    """One tunable value: what it means, and what counts as valid."""

    default: Any
    help: str
    kind: type = float
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None

    def validate(self, name: str, value: Any) -> Any:
        """Coerce ``value`` to this setting's type and check its range."""
        if self.kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}")
            return value
        if self.kind is str:
            value = str(value)
        else:
            try:
                value = self.kind(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name} must be {self.kind.__name__}, got {value!r}"
                ) from None
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"{name} must be one of {', '.join(map(str, self.choices))}, "
                f"got {value!r}"
            )
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{name} must be at least {self.minimum}, got {value}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{name} must be at most {self.maximum}, got {value}")
        return value


_COMMON: Dict[str, Setting] = {
    "epochs": Setting(100, "Passes over the training set.", int, minimum=1),
    "batch_size": Setting(
        64, "Samples per optimizer step. For SLEAP this is per GPU.",
        int, minimum=1,
    ),
    "learning_rate": Setting(5e-4, "Optimizer learning rate.", float, minimum=0),
    "save_every_n_epochs": Setting(
        20, "Checkpoint interval. Must be <= epochs or nothing is saved.",
        int, minimum=1,
    ),
    "validate_every_n_epochs": Setting(
        20, "Validation interval.", int, minimum=1,
    ),
}

# Bottom-up architectures only; see DLC3_PYTORCH_MODELS for why top-down and
# conditional-top-down networks are excluded.
DLC_ARCHITECTURES = (
    "cspnext_m", "cspnext_s", "cspnext_x",
    "dekr_w18", "dekr_w32", "dekr_w48",
    "dlcrnet_stride16_ms5", "dlcrnet_stride32_ms5",
    "hrnet_w18", "hrnet_w32", "hrnet_w48",
    "resnet_101", "resnet_50",
)

LP_BACKBONES = (
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnet50_animal_apose", "resnet50_animal_ap10k",
    "resnet50_human_jhmdb", "resnet50_human_res_rle",
    "resnet50_human_top_res", "resnet50_human_hand",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "vits_dino", "vits_dinov2", "vits_dinov3",
    "vitb_dino", "vitb_dinov2", "vitb_dinov3", "vitb_imagenet",
    "vitl_dinov3",
)

SLEAP_BACKBONES = (
    "unet", "unet_medium_rf", "unet_large_rf",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "swint_tiny", "swint_small", "swint_base",
)

TRAINING_SETTINGS: Dict[str, Dict[str, Setting]] = {
    "dlc": {
        **_COMMON,
        "network_architecture": Setting(
            "resnet_50", "Network to train.", str, choices=DLC_ARCHITECTURES,
        ),
        "train_fraction_percent": Setting(
            95, "Percent of labeled frames used for training.",
            float, minimum=1, maximum=99,
        ),
        "training_shuffle": Setting(1, "DLC shuffle index.", int, minimum=1),
        "max_snapshots_to_keep": Setting(
            5, "Most recent snapshots retained.", int, minimum=1,
        ),
        "rotation": Setting(30, "Rotation augmentation, degrees.", float, minimum=0),
        "scale_min": Setting(0.5, "Minimum scale augmentation.", float, minimum=0),
        "scale_max": Setting(1.25, "Maximum scale augmentation.", float, minimum=0),
        "crop_width": Setting(448, "Random crop width.", int, minimum=1),
        "crop_height": Setting(448, "Random crop height.", int, minimum=1),
        "motion_blur": Setting(True, "Apply motion-blur augmentation.", bool),
        "gaussian_noise": Setting(
            12.75, "Gaussian noise standard deviation.", float, minimum=0,
        ),
    },
    "lightning_pose": {
        **_COMMON,
        "learning_rate": Setting(1e-3, "Optimizer learning rate.", float, minimum=0),
        "backbone": Setting(
            "resnet50_animal_ap10k", "Backbone encoder.", str, choices=LP_BACKBONES,
        ),
        "horizontal_flip": Setting(False, "Random horizontal flip.", bool),
        "train_prob": Setting(
            0.95, "Training fraction; train_prob + val_prob must be <= 1.",
            float, minimum=0, maximum=1,
        ),
        "val_prob": Setting(
            0.05, "Validation fraction.", float, minimum=0, maximum=1,
        ),
        "unfreezing_epoch": Setting(
            20, "Epoch at which the backbone unfreezes.", int, minimum=0,
        ),
        "early_stopping": Setting(False, "Stop when validation plateaus.", bool),
        "early_stop_patience": Setting(
            3, "Epochs to wait before early stopping.", int, minimum=1,
        ),
        "dinov3_weights_dir": Setting(
            "",
            "Directory holding locally downloaded DINOv3 weights. Lightning "
            "Pose otherwise fetches them from a gated Hugging Face repo, which "
            "fails without credentials. Also settable via "
            "CHEESE3D_DINOV3_WEIGHTS.",
            str,
        ),
    },
    "sleap": {
        **_COMMON,
        "learning_rate": Setting(1e-4, "Optimizer learning rate.", float, minimum=0),
        "batch_size": Setting(
            4, "Samples per GPU. Effective batch is this times the GPU count.",
            int, minimum=1,
        ),
        "backbone": Setting(
            "convnext_tiny",
            "Backbone. UNet variants have no pretrained weights and tend to "
            "collapse on small datasets; prefer ConvNeXt or SwinT.",
            str, choices=SLEAP_BACKBONES,
        ),
        "output_stride": Setting(
            1,
            "Confidence-map stride; heatmaps are input_size/stride per "
            "keypoint, so each halving quadruples training cost. SLEAP-NN "
            "defaults to 1, which is ~26x the heatmap area DLC computes at "
            "its stride of 16.",
            int, choices=(1, 2, 4, 8, 16),
        ),
        "max_stride": Setting(
            0,
            "How far the encoder downsamples before the decoder upsamples "
            "back. 0 keeps the backbone's own default (UNet 16, ConvNeXt and "
            "SwinT 32). Larger means cheaper and coarser.",
            int, choices=(0, 8, 16, 32, 64),
        ),
        "stem_patch_stride": Setting(
            0,
            "ConvNeXt/SwinT patch-stem stride; 0 keeps SLEAP's default of 2. "
            "This is the dominant training cost on whole-frame data, because "
            "at 2 the first feature map stays at half input resolution.",
            int, choices=(0, 1, 2, 4),
        ),
        "filters": Setting(
            0, "UNet base filter count; 0 keeps the variant's default.",
            int, minimum=0,
        ),
        "filters_rate": Setting(
            0, "Filter growth per block; 0 keeps the backbone's default.",
            float, minimum=0,
        ),
        "convs_per_block": Setting(
            0, "Convolutions per block; 0 keeps the backbone's default.",
            int, minimum=0,
        ),
        "input_scale": Setting(
            1.0,
            "Input downscaling. SLEAP encodes whole frames, so 1.0 exhausts a "
            "48 GB GPU on 640x512 views at any batch size.",
            float, minimum=0.05, maximum=1.0,
        ),
        "validation_fraction_percent": Setting(
            10, "Percent of frames held out for validation.",
            float, minimum=1, maximum=50,
        ),
        "val_batch_size": Setting(4, "Validation batch size.", int, minimum=1),
        "optimizer": Setting(
            "Adam", "Optimizer.", str, choices=("Adam", "AdamW"),
        ),
        "min_steps_per_epoch": Setting(
            200,
            "Floor on optimizer steps per epoch. Above the natural batch count "
            "this repeats the dataset within one epoch; 0 means one pass.",
            int, minimum=0,
        ),
        "steps_per_epoch": Setting(
            0, "Exact steps per epoch; 0 uses the dataset length.", int, minimum=0,
        ),
        "save_top_k": Setting(
            1, "Best checkpoints retained by validation loss.", int, minimum=1,
        ),
        "save_last": Setting(True, "Also keep the final checkpoint.", bool),
        "early_stopping": Setting(True, "Stop when validation plateaus.", bool),
        "early_stop_patience": Setting(
            10, "Epochs to wait before early stopping.", int, minimum=1,
        ),
        "use_augmentation": Setting(True, "Apply geometric augmentation.", bool),
        "rotation_min": Setting(-15, "Minimum rotation, degrees.", float),
        "rotation_max": Setting(15, "Maximum rotation, degrees.", float),
        "scale_min": Setting(0.9, "Minimum scale.", float, minimum=0),
        "scale_max": Setting(1.1, "Maximum scale.", float, minimum=0),
        "translate": Setting(
            0.0, "Maximum translation, fraction of frame.", float, minimum=0, maximum=1,
        ),
        "peak_threshold": Setting(
            0.2,
            "Inference confidence threshold. Predictions below it become NaN.",
            float, minimum=0, maximum=1,
        ),
    },
}

# Lightning Pose accepts a whole imgaug dictionary, which is structured rather
# than scalar, so it is allowed through without per-key validation.
_FREEFORM: Dict[str, Tuple[str, ...]] = {"lightning_pose": ("imgaug",)}


def backend_settings(backend_type: str) -> Dict[str, Setting]:
    """Return the settings a backend accepts, keyed by name."""
    try:
        return TRAINING_SETTINGS[backend_type]
    except KeyError:
        known = ", ".join(sorted(TRAINING_SETTINGS))
        raise ValueError(
            f"Unknown backend {backend_type!r}; expected one of {known}"
        ) from None


def describe_training_settings(backend_type: str) -> str:
    """Render a backend's settings as a readable table."""
    schema = backend_settings(backend_type)
    width = max(len(name) for name in schema)
    lines = [f"{backend_type} training settings:"]
    for name, spec in schema.items():
        bounds = ""
        if spec.choices is not None:
            bounds = f" one of: {', '.join(map(str, spec.choices))}"
        elif spec.minimum is not None or spec.maximum is not None:
            low = "" if spec.minimum is None else f">= {spec.minimum:g}"
            high = "" if spec.maximum is None else f"<= {spec.maximum:g}"
            bounds = f" ({' and '.join(filter(None, (low, high)))})"
        lines.append(
            f"  {name:<{width}}  default {spec.default!r}{bounds}\n"
            f"  {'':<{width}}  {spec.help}"
        )
    return "\n".join(lines)


def validate_training_settings(backend_type: str, settings: Dict[str, Any],
                               fill_defaults: bool = False) -> Dict[str, Any]:
    """Check settings against a backend's schema, rejecting unknown keys.

    Args:
        backend_type: ``dlc``, ``lightning_pose`` or ``sleap``.
        settings: the values to check.
        fill_defaults: also return every unset setting at its default, which is
            useful for recording exactly what a run used.

    Raises:
        ValueError: on an unknown key, a wrong type, or an out-of-range value.
            An unknown key suggests the closest real one, since the usual cause
            is a typo or a setting borrowed from a different backend.
    """
    schema = backend_settings(backend_type)
    freeform = _FREEFORM.get(backend_type, ())
    checked: Dict[str, Any] = {}

    for name, value in settings.items():
        if name in freeform:
            checked[name] = value
            continue
        if name not in schema:
            hint = get_close_matches(name, list(schema) + list(freeform), n=1)
            suggestion = f"; did you mean {hint[0]!r}?" if hint else ""
            other = [b for b, s in TRAINING_SETTINGS.items()
                     if b != backend_type and name in s]
            if other and not hint:
                suggestion = f"; that setting belongs to {', '.join(other)}"
            raise ValueError(
                f"{backend_type} has no training setting {name!r}{suggestion}"
            )
        checked[name] = schema[name].validate(name, value)

    _check_relationships(backend_type, checked)

    if fill_defaults:
        return {name: checked.get(name, spec.default)
                for name, spec in schema.items()} | {
            name: checked[name] for name in freeform if name in checked
        }
    return checked


def _check_relationships(backend_type: str, settings: Dict[str, Any]) -> None:
    """Validate the constraints that span more than one setting."""
    def ordered(low: str, high: str) -> None:
        if low in settings and high in settings and settings[low] > settings[high]:
            raise ValueError(
                f"{low} ({settings[low]:g}) must not exceed {high} ({settings[high]:g})"
            )

    ordered("scale_min", "scale_max")
    if backend_type == "sleap":
        ordered("rotation_min", "rotation_max")
    if backend_type == "lightning_pose":
        # Lightning Pose asserts ckpt_every_n_epochs % check_val_every_n_epoch
        # == 0 and dies during config validation otherwise, after the model has
        # already been built.
        save = settings.get("save_every_n_epochs")
        validate = settings.get("validate_every_n_epochs")
        if save is not None and validate is not None and save % validate:
            raise ValueError(
                f"save_every_n_epochs ({save}) must be a multiple of "
                f"validate_every_n_epochs ({validate}) for Lightning Pose"
            )
        train, val = settings.get("train_prob"), settings.get("val_prob")
        if train is not None and val is not None and train + val > 1:
            raise ValueError(
                f"train_prob + val_prob must be at most 1, got {train + val:g}"
            )
    # A checkpoint interval longer than the run writes nothing at all, and the
    # tracking step then has no snapshot to load -- a failure that only shows
    # up much later, so catch it here.
    epochs = settings.get("epochs")
    for interval in ("save_every_n_epochs", "validate_every_n_epochs"):
        value = settings.get(interval)
        if epochs is not None and value is not None and value > epochs:
            raise ValueError(
                f"{interval} ({value}) exceeds epochs ({epochs}), so nothing "
                f"would be saved; lower it to {epochs} or fewer"
            )
