#!/usr/bin/env python3
"""
qc_widget.py — change‑based, bidirectional sync orchestrator
-------------------------------------------------------------
This script launches BOTH viewers with **data.py** as the single source of truth
for discovery, and then keeps them synchronized on frame changes.

Design goals (per project conventions):
  • No data parsing here — only orchestration & sync.
  • `rig_view.py` gets only: calibration, features CSV, annotation CSV, config.
  • `qc_video.py` gets only: videos_by_group (paths), calibration, one pose‑3d CSV,
    optional view name mapping & skeleton config.
  • `data.py` owns discovery and file conventions.

Usage:
    python qc_widget.py /path/to/dataset [--group GROUP_ID] [--config CONFIG_YAML] [--debug]

Hotkeys in BOTH windows: ← / → to step frames (controller‑driven).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import importlib.util
import sys
from typing import Dict, Optional

import numpy as np
import napari
from qtpy import QtCore

# ───────────────────────────────────────── helper: import sibling modules by path

def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ───────────────────────────────────────── sync controller
class _SyncController(QtCore.QObject):
    """Bridge RigViewer <-> QCReprojApp with a re‑entrancy guard and many fallbacks.

    Assumptions about APIs (best‑effort, optional fallbacks used where missing):
      • QC app exposes: .bus.frameChanged(int), .set_frame(int), .current_frame(), .max_frames()
      • RigViewer exposes: .show_frame(int) and/or .set_frame(int), a viewer with dims,
        and a plot widget with a signal among {currentFrameChanged, frameSelected, frameChanged}.
    """

    def __init__(self, rig, qc):
        super().__init__()
        self.rig = rig
        self.qc = qc
        self._guard = False

        # Discover total lengths (min guards clamping)
        self.Tq = self._guess_T_qc()
        self.Tr = self._guess_T_rig()
        self.Tmin = min([t for t in (self.Tq, self.Tr) if t is not None], default=None)

        # Listen to QC (explicit bus)
        if hasattr(self.qc, "bus") and hasattr(self.qc.bus, "frameChanged"):
            self.qc.bus.frameChanged.connect(self._from_qc)

        # Wrap RigViewer frame setters so *any* change triggers sync
        self._wrap_rig_frame_setters()

        # Also listen to RigViewer plot signals (critical for plot clicks)
        self._attach_rig_plot_signals()

        # Fallback listeners (slider / dims)
        self._attach_rig_fallback_listeners()

        # Key bindings in both windows
        @self.qc.viewer.bind_key("Left", overwrite=True)
        def _ql(_): self.step(-1)
        @self.qc.viewer.bind_key("Right", overwrite=True)
        def _qr(_): self.step(+1)
        try:
            @self.rig.viewer.bind_key("Left", overwrite=True)
            def _rl(_): self.step(-1)
            @self.rig.viewer.bind_key("Right", overwrite=True)
            def _rr(_): self.step(+1)
        except Exception:
            pass

        # Align starts
        self.goto(0)

    # ---------- Rig wrapping & listeners ----------
    def _wrap_rig_frame_setters(self):
        # Wrap show_frame
        if hasattr(self.rig, "show_frame") and callable(self.rig.show_frame):
            _orig_show = self.rig.show_frame
            def _wrapped_show(idx: int, *_a, **_k):
                res = _orig_show(int(idx), *_a, **_k)
                try:
                    cur = int(idx)
                except Exception:
                    cur = self._get_frame_rig()
                self._from_rig(cur)
                return res
            self.rig.show_frame = _wrapped_show  # type: ignore
        # Wrap set_frame
        if hasattr(self.rig, "set_frame") and callable(self.rig.set_frame):
            _orig_set = self.rig.set_frame
            def _wrapped_set(idx: int, *_a, **_k):
                res = _orig_set(int(idx), *_a, **_k)
                try:
                    cur = int(idx)
                except Exception:
                    cur = self._get_frame_rig()
                self._from_rig(cur)
                return res
            self.rig.set_frame = _wrapped_set  # type: ignore

    def _attach_rig_plot_signals(self):
        candidates = []
        for name in ("_plot_widget", "plot_widget", "_features_plot", "_tabs", "plot"):
            obj = getattr(self.rig, name, None)
            if obj is not None:
                candidates.append(obj)
            # try one level deeper
            for subname in ("plot", "widget"):
                sub = getattr(obj, subname, None) if obj is not None else None
                if sub is not None:
                    candidates.append(sub)
        signal_names = ("currentFrameChanged", "frameSelected", "frameChanged")
        for obj in candidates:
            for sig in signal_names:
                if hasattr(obj, sig):
                    try:
                        getattr(obj, sig).connect(lambda v, self=self: self._from_rig(int(v)))
                    except Exception:
                        pass

    def _attach_rig_fallback_listeners(self):
        # MagicGUI slider
        s = getattr(self.rig, "_frame_slider", None)
        if s is not None:
            try:
                s.changed.connect(lambda v: self._from_rig(int(v)))
            except Exception:
                pass
        # dims fallback
        try:
            self.rig.viewer.dims.events.current_step.connect(self._from_rig_dims)
        except Exception:
            pass

    # ---------- QC helpers ----------
    def _guess_T_qc(self) -> Optional[int]:
        try:
            return int(self.qc.max_frames())
        except Exception:
            try:
                for ly in self.qc.viewer.layers:
                    shp = getattr(ly.data, "shape", None)
                    if shp is not None and len(shp) >= 3:
                        return int(shp[0])
            except Exception:
                pass
            return None

    def _get_frame_qc(self) -> int:
        try:
            return int(self.qc.current_frame())
        except Exception:
            try:
                return int(self.qc.viewer.dims.current_step[0])
            except Exception:
                return 0

    def _set_frame_qc(self, fr: int):
        try:
            self.qc.set_frame(int(fr))
        except Exception:
            try:
                steps = list(self.qc.viewer.dims.current_step)
                steps[0] = int(fr)
                self.qc.viewer.dims.current_step = tuple(steps)
            except Exception:
                pass

    # ---------- Rig helpers ----------
    def _guess_T_rig(self) -> Optional[int]:
        try:
            fmin = int(getattr(self.rig, "_frame_min", 0))
            fmax = int(getattr(self.rig, "_frame_max"))
            return fmax - fmin + 1
        except Exception:
            pass
        try:
            s = self.rig._frame_slider
            return int(s.max - s.min + 1)
        except Exception:
            pass
        try:
            for ly in self.rig.viewer.layers:
                shp = getattr(ly.data, "shape", None)
                if shp is not None and len(shp) >= 3:
                    return int(shp[0])
        except Exception:
            pass
        return None

    def _get_frame_rig(self) -> int:
        for attr in ("_current", ):
            if hasattr(self.rig, attr):
                try:
                    return int(getattr(self.rig, attr))
                except Exception:
                    pass
        try:
            return int(self.rig._frame_slider.value)
        except Exception:
            pass
        try:
            return int(self.rig.viewer.dims.current_step[0])
        except Exception:
            return 0

    def _set_frame_rig(self, fr: int):
        for fn in ("show_frame", "set_frame"):
            if hasattr(self.rig, fn):
                try:
                    getattr(self.rig, fn)(int(fr))
                    return
                except Exception:
                    pass
        try:
            steps = list(self.rig.viewer.dims.current_step)
            steps[0] = int(fr)
            self.rig.viewer.dims.current_step = tuple(steps)
        except Exception:
            pass

    # ---------- generic helpers ----------
    def _clamp(self, fr: int) -> int:
        if self.Tmin is not None:
            return int(np.clip(fr, 0, self.Tmin - 1))
        return max(int(fr), 0)

    def goto(self, fr: int):
        fr = self._clamp(fr)
        self._guard = True
        try:
            self._set_frame_qc(fr)
            self._set_frame_rig(fr)
        finally:
            self._guard = False

    def step(self, delta: int):
        fr = self._get_frame_qc()  # QC as the truthy source
        self.goto(fr + int(delta))

    # ---------- slots ----------
    def _from_qc(self, fr: int):
        if self._guard:
            return
        fr = self._clamp(int(fr))
        self._guard = True
        try:
            self._set_frame_rig(fr)
        finally:
            self._guard = False

    def _from_rig(self, fr: int):
        if self._guard:
            return
        fr = self._clamp(int(fr))
        self._guard = True
        try:
            self._set_frame_qc(fr)
        finally:
            self._guard = False

    def _from_rig_dims(self, _event=None):
        if self._guard:
            return
        try:
            fr = int(self.rig.viewer.dims.current_step[0])
        except Exception:
            return
        self._from_rig(fr)


# ───────────────────────────────────────── CLI (pure orchestration)

def _build_inputs_for_rig(ds, config_override: Optional[Path]) -> Dict[str, Optional[Path]]:
    """Map dataset discovery to RigViewer kwargs."""
    calib = ds.calibration if getattr(ds, "calibration", None) else None
    features = getattr(ds, "features_csv", None)
    # choose one annotation CSV (pose‑3d); if multiple, take the first
    anno = None
    if getattr(ds, "pose3d_files", None):
        try:
            anno = Path(ds.pose3d_files[0])
        except Exception:
            # ds.pose3d_files may be objects with .path
            try:
                anno = Path(ds.pose3d_files[0].path)
            except Exception:
                anno = None
    cfg = config_override or (Path(ds.root) / "config.yaml")
    if not (isinstance(cfg, Path) and cfg.is_file()):
        cfg = None
    return {
        "calibration_path": Path(calib) if calib else None,
        "features_csv": Path(features) if features else None,
        "annotation_path": Path(anno) if anno else None,
        "config_path": cfg,
    }


def _build_inputs_for_qc(ds, group: Optional[str], config_override: Optional[Path]):
    """Map dataset discovery to QCReprojApp kwargs.

    videos_by_group is a dict: {group: {view_code: str_path}}, excluding calibration videos.
    """
    # videos
    videos_by_group = {}
    if getattr(ds, "videos", None):
        for g, vmap in ds.videos.items():
            videos_by_group[str(g)] = {}
            for view_code, p in vmap.items():
                p_str = str(p)
                # Exclude calibration clips (names containing 'cal')
                if "cal" in Path(p_str).name:
                    continue
                videos_by_group[str(g)][str(view_code)] = p_str
    if not videos_by_group:
        raise SystemExit("❌ No videos found. Check <root>/videos/<session>/* and config.yaml view codes.")

    # calibration
    if not getattr(ds, "calibration", None):
        raise SystemExit("❌ calibration file not found by data.py.")
    calibration_path = str(ds.calibration)

    # one pose‑3d CSV
    if not getattr(ds, "pose3d_files", None):
        raise SystemExit("❌ No pose_3d/*.csv found by data.py.")
    try:
        pose3d_csv = str(ds.pose3d_files[0])
    except Exception:
        pose3d_csv = str(ds.pose3d_files[0].path)

    # camera labels
    view_code_to_name = getattr(getattr(ds, "config", None), "view_code_to_name", {}) or {}

    # skeleton (optional)
    cfg = config_override or (Path(ds.root) / "config.yaml")
    skeleton_config = Path(cfg) if isinstance(cfg, Path) and cfg.is_file() else None

    return dict(
        videos_by_group=videos_by_group,
        calibration_path=calibration_path,
        pose3d_csv=pose3d_csv,
        view_code_to_name=view_code_to_name,
        group=group,
        skeleton_config=skeleton_config,
    )


def main():
    pa = argparse.ArgumentParser("Two‑window sync: RigViewer + QC reprojection (data.py‑powered)")
    pa.add_argument("dataset", type=Path, help="Dataset root folder")
    pa.add_argument("--group", type=str, default=None, help="Optional video group id")
    pa.add_argument("--config", type=Path, default=None, help="Override config.yaml for names/skeleton")
    pa.add_argument("--debug", action="store_true", help="Verbose discovery logs from data.py")
    args = pa.parse_args()

    dataset = args.dataset.resolve()
    if not dataset.exists() or not dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset}")

    here = Path(__file__).resolve().parent
    rig_mod = _import(here / "rig_view.py")
    qc_mod = _import(here / "qc_video.py")
    data_mod = _import(here / "data.py")

    ds = data_mod.discover_dataset(dataset, debug=args.debug)

    # Build inputs per new contracts
    rig_kwargs = _build_inputs_for_rig(ds, args.config)
    qc_kwargs  = _build_inputs_for_qc(ds, args.group, args.config)

    # Launch both apps (data‑only inputs; no parsing here)
    rig = rig_mod.RigViewer(
        calibration_path=rig_kwargs["calibration_path"],
        features_csv=rig_kwargs["features_csv"],
        annotation_path=rig_kwargs["annotation_path"],
        config_path=rig_kwargs["config_path"],
    )
    qc = qc_mod.QCReprojApp(
        videos_by_group=qc_kwargs["videos_by_group"],
        calibration_path=qc_kwargs["calibration_path"],
        pose3d_csv=qc_kwargs["pose3d_csv"],
        view_code_to_name=qc_kwargs["view_code_to_name"],
        group=qc_kwargs["group"],
        skeleton_config=qc_kwargs["skeleton_config"],
    )

    # Bridge & run
    _SyncController(rig, qc)
    napari.run()


if __name__ == "__main__":
    main()
