"""Cheese3D additions to Lightning Pose, applied without editing the package.

Everything here is a runtime patch of ``lightning_pose`` rather than a change
to its source, so the installed package stays exactly as published and these
additions survive a reinstall. Each one is written to be liftable into an
upstream pull request unchanged:

* ``vitl_dinov3`` -- Lightning Pose ships DINOv3 in Small and Base only, and
  ``build_backbone`` raises ``NotImplementedError`` for any other name. The
  Large checkpoint loads through exactly the same code path: the encoder width
  is read from the checkpoint's own config, so nothing downstream needs to know
  which size it got.

* Local DINOv3 weights -- the DINOv3 repos are gated on Hugging Face, so a
  machine without credentials cannot train those backbones even when the
  weights are already on disk. Redirecting the repo name to a directory is
  enough, because ``AutoModel.from_pretrained`` accepts a path.

The redirect is installed permanently for the process rather than around a
call. ``Model.from_dir`` is lazy: the backbone is not built until the model is
first used, well after any context manager around the constructor has exited.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

# Hugging Face repo -> directory name under the configured weights root.
DINOV3_MODELS = {
    "vits_dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb_dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl_dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

DINOV3_LOCAL_DIRECTORIES = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16_pretrain_lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3_vitb16_pretrain_lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3_vitl16_pretrain_lvd1689m",
}

# Backbones Lightning Pose does not define itself, added by this module.
ADDED_BACKBONES = ("vitl_dinov3",)


def resolve_local_weights(model_name: str, root: Path) -> Optional[Path]:
    """Return the local directory holding ``model_name``, if it is present."""
    folder = DINOV3_LOCAL_DIRECTORIES.get(model_name)
    candidate = root / folder if folder else root
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return candidate
    return None


def install(weights_dir: Optional[str] = None) -> Optional[Path]:
    """Add the Cheese3D backbones and local-weight loading to Lightning Pose.

    Idempotent, and safe to call from any process that is about to build a
    model -- training, inference workers, and checkpoint inspection all need
    it, because each rebuilds the backbone from scratch.

    Args:
        weights_dir: directory holding downloaded DINOv3 models. Falls back to
            ``CHEESE3D_DINOV3_WEIGHTS``.

    Returns:
        The weights directory in use, or None when only the extra backbone
        architectures were registered.
    """
    from lightning_pose.models.backbones import vits as vits_module

    weights_dir = weights_dir or os.environ.get("CHEESE3D_DINOV3_WEIGHTS")
    root = Path(weights_dir) if weights_dir else None

    if not getattr(vits_module, "_cheese3d_backbones_installed", False):
        _install_extra_backbones(vits_module)
        vits_module._cheese3d_backbones_installed = True

    if root is not None and getattr(
        vits_module, "_cheese3d_weights_root", None
    ) != str(root):
        _install_local_weights(vits_module, root)
        vits_module._cheese3d_weights_root = str(root)

    return root


def _install_local_weights(vits_module, root: Path) -> None:
    """Point DINOv3 repo names at ``root`` instead of the gated Hugging Face repo."""
    def load_from_local(model_name: str, pretrained_patch_size: int):
        local = resolve_local_weights(model_name, root)
        if local is None:
            raise RuntimeError(
                f"No local DINOv3 weights for {model_name} under {root}; "
                f"expected {root / DINOV3_LOCAL_DIRECTORIES.get(model_name, '')}"
                f"/config.json. Download it from "
                f"https://huggingface.co/{model_name}, or authenticate with "
                f"Hugging Face and unset CHEESE3D_DINOV3_WEIGHTS."
            )
        print(f"Lightning Pose loading DINOv3 weights from {local}", flush=True)
        return vits_module.VisionEncoderDino(
            model_name=str(local), pretrained_patch_size=pretrained_patch_size
        )

    vits_module._load_dinov3_with_auth_check = load_from_local


def _install_extra_backbones(vits_module) -> None:
    """Teach ``build_backbone`` the architectures Lightning Pose omits.

    Wraps rather than replaces: an unknown name still reaches the original
    implementation and raises its own ``NotImplementedError``, so this cannot
    mask a genuine typo.
    """
    original_build = vits_module.build_backbone

    def build_backbone(backbone_arch: str, **kwargs) -> Tuple[object, int]:
        if backbone_arch not in ADDED_BACKBONES:
            return original_build(backbone_arch, **kwargs)
        base = vits_module._load_dinov3_with_auth_check(
            model_name=DINOV3_MODELS[backbone_arch], pretrained_patch_size=16
        )
        # Same convention as every branch upstream: the pose head is sized from
        # the checkpoint's own width, so ViT-L's 1024 needs no special casing.
        return base, base.vision_encoder.config.hidden_size

    vits_module.build_backbone = build_backbone
