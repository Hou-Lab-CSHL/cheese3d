"""Run Anipose triangulation in an isolated JAX CUDA process."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _select_jax_backend(gpu: int, allow_cpu_fallback: bool) -> str:
    """Select one physical CUDA GPU before importing JAX or Aniposelib."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import jax

    try:
        # Requesting the backend explicitly exposes plugin/driver failures that
        # JAX's automatic backend selection would otherwise silently hide.
        gpu_devices = jax.devices("gpu")
    except Exception as error:
        gpu_devices = []
        print(f"JAX CUDA initialization failed: {error}", flush=True)
    if gpu_devices:
        # JAX sees the selected physical device as local device zero.
        print(f"JAX triangulation backend: GPU {gpu} ({gpu_devices[0]})", flush=True)
        return "gpu"
    if not allow_cpu_fallback:
        raise RuntimeError(
            "JAX CUDA is installed but no NVIDIA GPU is available. "
            "Enable triangulation.gpu_fallback_to_cpu or restore the driver."
        )
    print("WARNING: JAX GPU unavailable; triangulation is falling back to CPU.",
          flush=True)
    return "cpu"


def main() -> None:
    """Load an Anipose configuration and triangulate all or one session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--session", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    _select_jax_backend(args.gpu, args.allow_cpu_fallback)
    # Import Anipose only after CUDA visibility is fixed; Aniposelib otherwise
    # initializes JAX on every visible device during its first triangulation.
    from anipose.anipose import load_config
    from anipose.triangulate import process_session, triangulate_all

    config = load_config(str(args.config))
    if args.session:
        process_session(config, args.session)
    else:
        triangulate_all(config)


if __name__ == "__main__":
    main()
