#!/usr/bin/env python3
"""
rig_view.py
===========
Rig viewer + Cheese3D features plot + 3-D annotations (data discovered in main() via data.py).

Design:
  • Discovery is centralized in data.py. main() calls discover_dataset() and
    build_rig_view_inputs(), then passes ONLY paths to RigViewer.
  • RigViewer is data-only and accepts:
        - calibration_path (required; toml/json)
        - features_csv (optional)
        - annotation_path (optional; pose-3d CSV; visualized if present)
        - config_path (optional; for 'skeleton' edges)
  • Left/Right keys step frames. Plot ⇄ viewer stay synced.
  • Napari "Layer list" and "Layer controls" hidden on launch (toggle 'L').
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import cv2
import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QDockWidget  # robust fallback for dock toggling

# Optional plotting widget for cheese3d_features.csv
from cheese3d_annotator.data_visualizer.plot import FeaturesPlotWidget

_COLORMAP = {
    'nose(bottom)': [0.0039, 0.4510, 0.6980, 1.0],
    'nose(tip)': [0.0039, 0.4510, 0.6980, 1.0],
    'nose(top)': [0.1569, 0.4704, 0.5783, 1.0],
    'pad(top)(left)': [0.7992, 0.5517, 0.0755, 1.0],
    'pad(side)(left)': [0.7589, 0.5684, 0.0754, 1.0],
    'pad(center)': [0.6067, 0.5788, 0.1516, 1.0],
    'pad(top)(right)': [0.0468, 0.6078, 0.4298, 1.0],
    'pad(side)(right)': [0.3680, 0.5104, 0.2547, 1.0],
    'lowerlip': [0.8291, 0.3866, 0.1301, 1.0],
    'upperlip(left)': [0.8216, 0.4082, 0.2862, 1.0],
    'upperlip(right)': [0.8154, 0.4262, 0.4163, 1.0],
    'eye(front)(left)': [0.7976, 0.5006, 0.6281, 1.0],
    'eye(top)(left)': [0.7965, 0.5144, 0.5777, 1.0],
    'eye(back)(left)': [0.7957, 0.5248, 0.5399, 1.0],
    'eye(bottom)(left)': [0.7946, 0.5386, 0.4896, 1.0],
    'eye(front)(right)': [0.8758, 0.6198, 0.6040, 1.0],
    'eye(top)(right)': [0.9029, 0.6364, 0.6765, 1.0],
    'eye(back)(right)': [0.9233, 0.6489, 0.7309, 1.0],
    'eye(bottom)(right)': [0.9504, 0.6655, 0.8035, 1.0],
    'ear(base)(left)': [0.7562, 0.6265, 0.7170, 1.0],
    'ear(top)(left)': [0.6992, 0.6115, 0.6727, 1.0],
    'ear(tip)(left)': [0.6564, 0.6003, 0.6394, 1.0],
    'ear(bottom)(left)': [0.5994, 0.5854, 0.5952, 1.0],
    'ear(base)(right)': [0.8199, 0.7900, 0.3164, 1.0],
    'ear(top)(right)': [0.8687, 0.8326, 0.2627, 1.0],
    'ear(tip)(right)': [0.9052, 0.8646, 0.2224, 1.0],
    'ear(bottom)(right)': [0.8770, 0.8678, 0.2588, 1.0],
    'ref(head-post)': [0.4411, 0.7370, 0.7878, 1.0],
}

# ─────────────────────────────────────────────── helpers (local, tiny IO)
def _load_calibration_dict(calib_path: Path) -> dict:
    """Read TOML/JSON calibration into a dict. Tolerant to either format."""
    if not calib_path or not calib_path.is_file():
        return {}
    try:
        if calib_path.suffix.lower() == ".toml":
            try:
                import tomllib  # py311+
            except Exception:
                import tomli as tomllib  # fallback
            return tomllib.loads(calib_path.read_text())
        if calib_path.suffix.lower() == ".json":
            import json
            return json.loads(calib_path.read_text())
    except Exception:
        pass
    return {}

def _load_cheese3d_features(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path and path.is_file():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def _load_skeleton_edges(config_path: Optional[Path]) -> List[Tuple[str, str]]:
    if not config_path or not config_path.is_file():
        return []
    try:
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        edges = cfg.get("skeleton", []) or []
        out: List[Tuple[str, str]] = []
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                out.append((str(e[0]), str(e[1])))
        return out
    except Exception:
        return []

def _rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    Rm, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return Rm

def _imgplane_corners(K: np.ndarray, depth: float, size: Tuple[int, int]):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = size
    pix = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    X = np.empty((4, 3))
    X[:, 0] = (pix[:, 0] - cx) * depth / fx
    X[:, 1] = (pix[:, 1] - cy) * depth / fy
    X[:, 2] = depth
    return X

def _parse_keypoint_bases(columns: List[str]) -> List[str]:
    cols = set(columns)
    bases: List[str] = []
    for c in cols:
        if c.endswith("_x"):
            b = c[:-2]
            if f"{b}_y" in cols and f"{b}_z" in cols:
                bases.append(b)
    bases.sort()
    return bases

def _extract_head2world(row) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        have_M = all(f"M_{i}{j}" in row for i in range(3) for j in range(3))
        have_C = all(k in row for k in ("center_0", "center_1", "center_2"))
        if not (have_M and have_C):
            return None
        R_wh = np.array([[row[f"M_{i}{j}"] for j in range(3)] for i in range(3)], float)
        c_h  = np.array([row["center_0"], row["center_1"], row["center_2"]], float)
        return R_wh, c_h
    except Exception:
        return None

def _apply_head2world_if_present(Xh: np.ndarray, xform: Optional[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Mirror qc_video/rig_view: X_world = R_wh.T @ (X_head + c_h)
    """
    if Xh.size == 0 or xform is None:
        return Xh
    R_wh, c_h = xform
    return (R_wh.T @ (Xh + c_h).T).T


# ─────────────────────────────────────────────── main viewer (data-only)
class RigViewer:
    _LAYER_DOCK_TITLES = ("Layer list", "Layer controls", "Layers")
    _UNWANTED = ["ref(*)"]  # glob-like; filter names such as ref(0), ref(...)

    def __init__(
        self,
        calibration_path: Path,
        features_csv: Optional[Path] = None,
        annotation_path: Optional[Path] = None,  # pose-3d CSV
        skeleton_config: Optional[Path | List[Tuple[str, str]]] = None
    ):
        self.calib_path = calibration_path
        self.features_csv = features_csv
        self.annotation_path = annotation_path

        # Load calibration dict (toml/json)
        self._calib_dict = _load_calibration_dict(self.calib_path)
        if not self._calib_dict:
            raise FileNotFoundError(f"Failed to read calibration: {self.calib_path}")

        # Optional features dataframe (drives plot)
        self._features_df = _load_cheese3d_features(self.features_csv)

        # Optional 3-D annotations dataframe (pose-3d)
        self._anno_df: Optional[pd.DataFrame] = None
        self._anno_bases: List[str] = []
        if self.annotation_path and self.annotation_path.is_file():
            try:
                df = pd.read_csv(self.annotation_path)
                # sort by 'frame' if present
                if "frame" in df.columns:
                    df = df.sort_values("frame").reset_index(drop=True)
                self._anno_df = df
                self._anno_bases = _parse_keypoint_bases(df.columns.tolist())
            except Exception as e:
                print(f"[warn] Could not load annotations: {e}")
                self._anno_df = None
                self._anno_bases = []

        # Skeleton edges (optional)
        if isinstance(skeleton_config, list):
            self._skeleton_edges = skeleton_config
        else:
            self._skeleton_edges = _load_skeleton_edges(skeleton_config)

        # Frame range: prefer annotations, else features, else 0
        self._frame_min = 0
        if self._anno_df is not None and len(self._anno_df) > 0:
            self._frame_max = len(self._anno_df) - 1
        elif self._features_df is not None and len(self._features_df) > 0:
            self._frame_max = len(self._features_df) - 1
        else:
            self._frame_max = 0

        # napari (3-D rig)
        self.viewer = napari.Viewer(ndisplay=3, title="Rig Viewer")
        qtv = self.viewer.window.qt_viewer
        getattr(qtv, "_dockLayerList", qtv.dockLayerList).setVisible(False)
        getattr(qtv, "_dockLayerControls", qtv.dockLayerControls).setVisible(False)
        self._add_camera_rig_from_dict(self._calib_dict)

        # Hide the built-in Layers UI (layer list + layer controls). Toggle with 'L'
        self._set_layers_ui_visible(False)

        # dynamic layers for annotations
        self._points_layer = None
        self._skel_layer = None

        # keyboard shortcuts
        self._current = self._frame_min

        @self.viewer.bind_key('Left', overwrite=True)
        def _go_left(_):
            self.show_frame(self._current - 1)

        @self.viewer.bind_key('Right', overwrite=True)
        def _go_right(_):
            self.show_frame(self._current + 1)

        @self.viewer.bind_key('L', overwrite=True)
        def _toggle_layers_ui(_):
            self._toggle_layers_ui()

        # bottom dock: Cheese3D features plot (if given)
        self._plot_widget = None
        if self.features_csv and self.features_csv.is_file():
            try:
                self._plot_widget = FeaturesPlotWidget(self.features_csv)
                self.viewer.window.add_dock_widget(
                    self._plot_widget, area="bottom", name="Cheese3D Features"
                )
                # plot → viewer (support a few possible signal names)
                for sig in ("currentFrameChanged", "frameSelected", "frameChanged"):
                    if hasattr(self._plot_widget, sig):
                        getattr(self._plot_widget, sig).connect(self.show_frame)
                        break
            except Exception as e:
                print(f"[warn] Could not create FeaturesPlotWidget: {e}")

        # initial frame
        self.show_frame(self._frame_min)

    # ── Layers UI visibility control ────────────────────────────────────
    def _set_layers_ui_visible(self, visible: bool) -> None:
        try:
            win_menu = self.viewer.window.window_menu
            for act in win_menu.actions():
                text = (act.text() or "").strip()
                if any(lbl.lower() in text.lower() for lbl in self._LAYER_DOCK_TITLES):
                    if act.isChecked() != visible:
                        act.trigger()
        except Exception:
            try:
                qtwin = self.viewer.window._qt_window
                for qdock in qtwin.findChildren(QDockWidget):
                    title = (qdock.windowTitle() or "").strip().lower()
                    if title in {t.lower() for t in self._LAYER_DOCK_TITLES}:
                        qdock.setVisible(visible)
            except Exception as e2:
                print(f"[warn] Could not change Layers UI visibility: {e2}")

    def _layers_ui_currently_visible(self) -> bool:
        try:
            win_menu = self.viewer.window.window_menu
            for act in win_menu.actions():
                text = (act.text() or "").strip().lower()
                if any(lbl.lower() in text for lbl in self._LAYER_DOCK_TITLES):
                    if act.isChecked():
                        return True
        except Exception:
            pass
        try:
            qtwin = self.viewer.window._qt_window
            for qdock in qtwin.findChildren(QDockWidget):
                title = (qdock.windowTitle() or "").strip().lower()
                if title in {t.lower() for t in self._LAYER_DOCK_TITLES} and qdock.isVisible():
                    return True
        except Exception:
            pass
        return False

    def _toggle_layers_ui(self) -> None:
        self._set_layers_ui_visible(not self._layers_ui_currently_visible())

    # ── public API ──────────────────────────────────────────────────────
    def show_frame(self, idx: int) -> None:
        idx = int(np.clip(idx, self._frame_min, self._frame_max))
        self._current = idx

        # update 3-D annotations
        self._update_annotations(idx)

        # keep plot in sync (viewer → plot)
        if self._plot_widget is not None:
            for fn in ("goto_frame", "set_frame", "setCurrentFrame"):
                try:
                    if hasattr(self._plot_widget, fn):
                        getattr(self._plot_widget, fn)(self._current)
                        break
                except Exception:
                    pass

    # ── static rig from calibration dict ────────────────────────────────
    def _add_camera_rig_from_dict(self, calib: dict, frustum_z: float = 20.0) -> None:
        # Accept either cheese3d-style top-level dict or nested dicts; look for sections with 'matrix'
        cams = {k: v for k, v in calib.items() if isinstance(v, dict) and "matrix" in v}
        if not cams:
            # fallback: if top-level is like {"camX": {...}}, keep those dicts
            cams = {k: v for k, v in calib.items() if isinstance(v, dict)}

        if not cams:
            print("[warn] No camera sections found in calibration file.")
            return

        frusta = []
        for cam in cams.values():
            try:
                K = np.asarray(cam.get("matrix") or cam.get("K") or cam.get("camera_matrix")).reshape(3, 3)
                rvec = np.asarray(cam.get("rotation") or cam.get("rvec")).reshape(3)
                tvec = np.asarray(cam.get("translation") or cam.get("tvec")).reshape(3)
                Rm = _rvec_to_R(rvec)
                C = -(Rm.T @ tvec)
                size = tuple(cam.get("size") or (640, 480))
            except Exception as e:
                print(f"[warn] Bad camera entry skipped: {e}")
                continue

            imgplane = _imgplane_corners(K, frustum_z, size)
            imgplane_w = (Rm.T @ imgplane.T).T + C
            for i in range(4):
                frusta.append(np.stack([C, imgplane_w[i]]))
                frusta.append(np.stack([imgplane_w[i], imgplane_w[(i + 1) % 4]]))

        if frusta:
            self.viewer.add_shapes(
                np.asarray(frusta, float),
                shape_type="path",
                edge_color="cyan",
                edge_width=0.2,
                name="Cam frusta",
                opacity=0.5,
            )

    # ── 3-D annotations (points + optional skeleton) ────────────────────
    def _update_annotations(self, frame_idx: int) -> None:
        if self._anno_df is None or self._anno_df.empty:
            # nothing to show; hide layers if they exist
            if self._points_layer is not None:
                self._points_layer.visible = False
            if self._skel_layer is not None:
                self._skel_layer.visible = False
            return

        if frame_idx < 0 or frame_idx >= len(self._anno_df):
            return

        row = self._anno_df.iloc[frame_idx]

        # gather xyz for every base present in this row
        names, pts = [], []
        for b in self._anno_bases:
            try:
                x, y, z = row[f"{b}_x"], row[f"{b}_y"], row[f"{b}_z"]
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    names.append(b)
                    pts.append([float(x), float(y), float(z)])
            except Exception:
                continue

        if not pts:
            if self._points_layer is not None:
                self._points_layer.visible = False
            if self._skel_layer is not None:
                self._skel_layer.visible = False
            return

        # filter unwanted names (glob-ish)
        def _unwanted(n: str) -> bool:
            for pat in self._UNWANTED:
                # convert 'ref(*)' to regex
                rx = "^" + re.escape(pat).replace("\\*", ".*") + "$"
                if re.fullmatch(rx, n):
                    return True
            return False

        keep = [not _unwanted(n) for n in names]
        names = [n for n, k in zip(names, keep) if k]
        Xh = np.asarray([p for p, k in zip(pts, keep) if k], float)

        # optional head→world transform per row
        Xw = _apply_head2world_if_present(Xh, _extract_head2world(row))

        colors = np.array([_COLORMAP.get(n, [1.0, 1.0, 1.0, 1.0]) for n in names], dtype=float)

        # points layer
        if self._points_layer is None:
            self._points_layer = self.viewer.add_points(
                Xw,
                size=1.0,
                face_color=colors,
                name="3-D annotations",
                properties={"label": np.asarray(names, dtype=object)},
                text={"string": "{label}", "visible": False},
            )
        else:
            self._points_layer.data = Xw
            self._points_layer.face_color = colors
            self._points_layer.properties = {"label": np.asarray(names, dtype=object)}
            self._points_layer.visible = True

        # optional skeleton
        segs = []
        if self._skeleton_edges and len(names) > 1:
            name2idx = {n: i for i, n in enumerate(names)}
            for a, b in self._skeleton_edges:
                ia = name2idx.get(a); ib = name2idx.get(b)
                if ia is None or ib is None:
                    continue
                pa, pb = Xw[ia], Xw[ib]
                if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                    continue
                segs.append(np.stack([pa, pb], axis=0))
        if segs:
            data = np.asarray(segs, float)
            if self._skel_layer is None:
                self._skel_layer = self.viewer.add_shapes(
                    data,
                    shape_type="path",
                    edge_color="white",
                    edge_width=0.15,
                    name="Skeleton",
                )
            else:
                self._skel_layer.data = data
                self._skel_layer.visible = True
        elif self._skel_layer is not None:
            self._skel_layer.visible = False


# ─────────────────────────────────────────────── CLI (all discovery here)
def main() -> None:
    pa = argparse.ArgumentParser("Napari rig viewer + Cheese3D features plot + 3-D annotations (data.py-powered)")
    pa.add_argument("dataset", type=Path, help="Dataset root folder")
    pa.add_argument("--calib", type=Path, help="Override calibration.toml/json")
    pa.add_argument("--features", type=Path, help="Override cheese3d_features.csv for the plot")
    pa.add_argument("--annotation", type=Path, help="Override pose-3d CSV for 3-D annotations")
    pa.add_argument("--config", type=Path, help="Optional config.yaml (for 'skeleton' edges)")
    pa.add_argument("--debug", action="store_true", help="Verbose discovery logs from data.py")
    args = pa.parse_args()

    if not args.dataset.exists() or not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset}")

    # 1) Discover everything with data.py (single source of truth)
    ds = discover_dataset(args.dataset, debug=args.debug)
    rv_inputs = build_rig_view_inputs(ds)
    # rv_inputs keys: calibration, features_csv, annotation, view_code_to_name

    # 2) Respect CLI overrides when provided
    calib_path = args.calib or (Path(rv_inputs["calibration"]) if rv_inputs.get("calibration") else None)
    if calib_path is None or not Path(calib_path).is_file():
        raise FileNotFoundError("calibration file not found (use --calib to specify explicitly).")

    features_csv = args.features or (Path(rv_inputs["features_csv"]) if rv_inputs.get("features_csv") else None)
    if features_csv is not None and not Path(features_csv).is_file():
        print(f"[warn] features csv not found at {features_csv}; continuing without plot.")
        features_csv = None

    annotation_path = args.annotation or (Path(rv_inputs["annotation"]) if rv_inputs.get("annotation") else None)
    if annotation_path is not None and not Path(annotation_path).is_file():
        print(f"[warn] annotation file not found at {annotation_path}; ignoring.")
        annotation_path = None

    # Prefer explicit --config; else, try dataset/config.yaml
    config_path = args.config if args.config else (args.dataset / "config.yaml")
    if not (config_path.is_file() if isinstance(config_path, Path) else False):
        config_path = None

    # 3) Launch data-only viewer
    _ = RigViewer(
        calibration_path=Path(calib_path),
        features_csv=Path(features_csv) if features_csv else None,
        annotation_path=Path(annotation_path) if annotation_path else None,
        config_path=Path(config_path) if config_path else None,
    )
    napari.run()


if __name__ == "__main__":
    main()
