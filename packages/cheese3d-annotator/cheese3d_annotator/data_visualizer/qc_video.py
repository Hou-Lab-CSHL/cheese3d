#!/usr/bin/env python3
"""
qc_video.py  — minimal QC viewer for facial keypoints (2-D overlays + skeletons)
--------------------------------------------------------------------------------------
Focus:
  1) Load 3D keypoints from pose_3d/*.csv
  2) Optionally apply per-frame head->world transform (M_ij, center_k)
  3) Load calibration and project 3D -> 2D per camera
  4) Visualize video frames with overlaid 2D keypoints & skeletons in a tiled grid
  5) Filter out unwanted keypoints by name patterns (e.g., 'ref(*)')
"""

from __future__ import annotations
import sys, re, math, json, warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from fnmatch import fnmatch

import numpy as np
import cv2
import napari
from napari_video.napari_video import VideoReaderNP
from qtpy import QtCore, QtWidgets

# Quiet benign warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Camera parsing / ordering
_CAM_RE = re.compile(r"_([A-Z]{1,2})(?=_|\.)")
# Stable order for tiling (fallback when config doesn't specify)
_CAM_ORDER = {"TL": 0, "TC": 1, "TR": 2, "L": 3, "BC": 4, "R": 5}

# Unwanted keypoint name patterns (glob-style). Example: "ref(*)" drops ref(0), ref(anything)...
_UNWANTED = ["ref(*)"]

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

def _is_unwanted(name: str) -> bool:
    return any(fnmatch(name, pat) for pat in _UNWANTED)

# -------------------- Calibration load & 3D->2D projection --------------------

# tomllib (3.11+) else tomli
try:
    import tomllib  # type: ignore
except Exception:
    import tomli as tomllib  # type: ignore


def embed_qc_into_rig(rig, qc, area="right", name="QC Back-Projection"):
    """
    Reparents qc.viewer's central widget into rig.viewer as a dock widget.
    Leaves all layers, keybindings, and signals intact. No data logic touched.
    """
    rig_win = rig.viewer.window
    qc_win  = qc.viewer.window

    qmain_qc: QtWidgets.QMainWindow = qc_win._qt_window
    central  = qmain_qc.centralWidget()
    if central is None:
        return

    qmain_qc.setCentralWidget(None)
    try:
        rig_win.add_dock_widget(central, area=area, name=name)
    except Exception:
        dock = QtWidgets.QDockWidget(name, rig_win._qt_window)
        dock.setObjectName(f"{name} (dock)")
        dock.setWidget(central)
        rig_win._qt_window.addDockWidget(getattr(QtCore, f"{area.capitalize()}DockWidgetArea"), dock)

    try:
        rig.viewer.window._qt_window.raise_()
        rig.viewer.window._qt_window.activateWindow()
    except Exception:
        pass


def _load_calibration_raw(calib_path: Path) -> Dict:
    """Return raw calibration dict from TOML or JSON."""
    if calib_path.suffix.lower() == ".toml":
        return tomllib.loads(calib_path.read_text())
    elif calib_path.suffix.lower() == ".json":
        return json.loads(calib_path.read_text())
    raise ValueError(f"Unsupported calibration format: {calib_path.suffix}")


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


def _scale_K_for_video(K: np.ndarray, calib_size: Tuple[float, float], video_size: Tuple[int, int]) -> np.ndarray:
    """Scale intrinsics to the given video size."""
    calib_w, calib_h = calib_size
    vid_w, vid_h     = float(video_size[0]), float(video_size[1])
    sx, sy = vid_w / float(calib_w), vid_h / float(calib_h)
    K = np.asarray(K, float).copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def _normalize_cam_name(name: str) -> str:
    """Return a short camera code-like key for matching (e.g. 'cam_BC' -> 'BC')."""
    name = name.strip()
    m = _CAM_RE.search(name)
    if m:
        return m.group(1)
    tokens = re.split(r"[^A-Za-z0-9]+", name)
    for t in reversed(tokens):
        if t in {"TL", "TC", "TR", "L", "BC", "R"}:
            return t
    return name


def _build_calib_map(raw: Dict, video_size: Tuple[int, int]) -> Dict[str, dict]:
    """
    Accept Cheese3D TOML-style or JSON-style structures.
    Build {cam_code: {'K','dist','rvec','tvec'}}; scale K to video_size.
    """
    if any(isinstance(v, dict) and "matrix" in v for v in raw.values()):
        cams = {v.get("name", k): v for k, v in raw.items() if isinstance(v, dict) and "matrix" in v}
    else:
        cams = raw

    first = next((v for v in cams.values() if isinstance(v, dict)), None)
    if first is not None:
        calib_sz = np.array(first.get("size", [video_size[0], video_size[1]]), float)
    else:
        calib_sz = np.array([video_size[0], video_size[1]], float)

    out: Dict[str, dict] = {}
    for name, v in cams.items():
        if not isinstance(v, dict):
            continue
        K    = v.get("matrix") or v.get("K") or v.get("camera_matrix")
        rvec = v.get("rotation") or v.get("rvec")
        tvec = v.get("translation") or v.get("tvec")
        dist = v.get("distortions", v.get("distortion", []))
        if K is None or rvec is None or tvec is None:
            continue
        K    = np.asarray(K, float).reshape(3, 3)
        rvec = np.asarray(rvec, float).reshape(3)
        tvec = np.asarray(tvec, float).reshape(3)
        dist = np.asarray(dist, float).reshape(-1) if dist is not None else np.zeros((0,), float)

        K = _scale_K_for_video(K, calib_sz, video_size)
        cam_code = _normalize_cam_name(str(name))
        out[cam_code] = {"K": K, "dist": dist, "rvec": rvec, "tvec": tvec, "raw_name": str(name)}
    return out


def _project_pts(X_world: np.ndarray, prm: dict) -> np.ndarray:
    """Project 3D -> 2D using cv2.projectPoints."""
    if X_world.size == 0:
        return np.zeros((0, 2), float)
    p, _ = cv2.projectPoints(X_world.reshape(-1, 3), prm["rvec"], prm["tvec"], prm["K"], prm["dist"])
    return p.reshape(-1, 2)

# ----------------------------- Keypoints CSV & colors --------------------------

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


def _load_keypoints_csv_with_xforms(csv_path: Path):
    """
    Returns:
      bases:           ordered list of keypoint base names
      fr2X_head:       frame -> (N,3) head-space (or world if no xform)
      fr2names:        frame -> [names for the rows in fr2X_head]
      fr2xform:        frame -> (R_wh, c_h) if present
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "frame" in df.columns:
        df = df.sort_values("frame").reset_index(drop=True)
        frames = df["frame"].astype(int).to_numpy()
    else:
        frames = np.arange(len(df), dtype=int)

    bases = _parse_keypoint_bases(df.columns.tolist())

    fr2X_head: dict[int, np.ndarray] = {}
    fr2names:  dict[int, List[str]] = {}
    fr2xform:  dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for i, fr in enumerate(frames):
        row = df.iloc[i]
        pts, names = [], []
        for b in bases:
            x, y, z = row[f"{b}_x"], row[f"{b}_y"], row[f"{b}_z"]
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                pts.append([float(x), float(y), float(z)])
                names.append(b)
        Xh = np.asarray(pts, float) if pts else np.zeros((0, 3), float)
        fr2X_head[int(fr)] = Xh
        fr2names[int(fr)]  = names
        xform = _extract_head2world(row)
        if xform is not None:
            fr2xform[int(fr)] = xform

    return bases, fr2X_head, fr2names, fr2xform


def _apply_head2world_if_present(Xh: np.ndarray, xform: Optional[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Mirror rig_view: X_world = R_wh.T @ (X_head + c_h)
    """
    if Xh.size == 0 or xform is None:
        return Xh
    R_wh, c_h = xform
    return (R_wh.T @ (Xh + c_h).T).T


def _make_color_map(bases: List[str]) -> Dict[str, np.ndarray]:
    """
    Match rig_view.py behavior: MPL 'turbo' resampled(len(bases)), assign by index.
    Returns name -> (R,G,B) in 0..1 floats.
    """
    if len(bases) <= 0:
        return {}
    colors = np.array([_COLORMAP.get(n, [1.0, 1.0, 1.0, 1.0]) for n in bases], dtype=float)
    return {name: colors[i] for i, name in enumerate(bases)}

# --------------------------- External frame bus (NEW) --------------------------

class _FrameBus(QtCore.QObject):
    """Signal bus for robust external frame synchronization."""
    frameChanged = QtCore.Signal(int)  # emitted whenever the app changes to a new frame

    def __init__(self):
        super().__init__()

# --------------------------------- App (data-only) ----------------------------------------

class QCReprojApp:
    """
    Data-only viewer. All discovery must happen in main() via data.py.

    Parameters
    ----------
    videos_by_group : Dict[str, Dict[str, str]]
        {group_id: {view_code: "/path/to/video"}}
    calibration_path : Path | str
    pose3d_csv : Path | str
    view_code_to_name : Optional[Dict[str, str]]
    group : Optional[str]
        Which group to open initially (must be a key in videos_by_group)
    skeleton_config : Optional[Path]
        If provided, read 'skeleton' edges from this YAML (usually config.yaml)
    """
    def __init__(
        self,
        videos_by_group: Dict[str, Dict[str, str]],
        calibration_path: Path | str,
        pose3d_csv: Path | str,
        view_code_to_name: Optional[Dict[str, str]] = None,
        group: Optional[str] = None,
        skeleton_config: Optional[Path | List[Tuple[str, str]]] = None,
    ):
        if not videos_by_group:
            sys.exit("❌ videos_by_group is empty.")
        self.videos_by_group = videos_by_group

        self.calib_path = Path(calibration_path)
        if not self.calib_path.exists():
            sys.exit(f"❌ calibration not found: {self.calib_path}")

        self.pose3d_csv = Path(pose3d_csv)
        if not self.pose3d_csv.exists():
            sys.exit(f"❌ pose_3d CSV not found: {self.pose3d_csv}")

        self.view_code_to_name = view_code_to_name or {}

        # Choose group & camera ordering (prefer config order)
        self.group_id = group if (group and group in videos_by_group) else next(iter(videos_by_group.keys()))
        grp_videos = videos_by_group[self.group_id]  # {view_code: path}
        cfg_codes = list(self.view_code_to_name.keys())
        if cfg_codes:
            ordered_codes = [c for c in cfg_codes if c in grp_videos] + [c for c in grp_videos if c not in cfg_codes]
        else:
            ordered_codes = sorted(grp_videos.keys(), key=lambda c: _CAM_ORDER.get(c, 999))
        self.cam_codes = ordered_codes
        self.vids = [Path(grp_videos[c]) for c in self.cam_codes]
        if not self.vids:
            sys.exit("❌ Selected group has no videos.")

        # Load 3D points (and optional per-frame transforms)
        self.bases, self.X_head_per_frame, self.names_per_frame, self.xform_per_frame = _load_keypoints_csv_with_xforms(self.pose3d_csv)
        self.name2color = _make_color_map(self.bases)

        # Skeleton edges (optional)
        if isinstance(skeleton_config, list):
            self.skeleton_edges = skeleton_config
        else:
            self.skeleton_edges = _load_skeleton_edges(skeleton_config)

        # Open video readers; gather sizes & timeline length
        self.readers: dict[str, VideoReaderNP] = {}
        counts = []
        ref_w = ref_h = None
        for code, vpath in zip(self.cam_codes, self.vids):
            rdr = VideoReaderNP(str(vpath))
            self.readers[code] = rdr
            # try to read frame count from reader; else from cv2 fallback
            n = getattr(rdr, "n_frames", None) or getattr(getattr(rdr, "_reader", None), "n_frames", None)
            if n is None:
                _, _, _, n = self._video_props(vpath)
            counts.append(int(n))
            if ref_w is None or ref_h is None:
                shp = getattr(rdr, "shape", None)
                if shp is not None and len(shp) >= 3:
                    ref_h, ref_w = int(shp[1]), int(shp[2])
        if ref_w is None or ref_h is None:
            ref_w, ref_h, _, _ = self._video_props(self.vids[0])
        self.video_size = (ref_w, ref_h)
        self.T = int(min(counts)) if counts else 1

        # Calibration (scaled to video size) with tolerant camera naming
        raw_cal = _load_calibration_raw(self.calib_path)
        self.calib_map = _build_calib_map(raw_cal, self.video_size)   # keyed by short cam code ('BC','L',...)

        # Napari viewer & layers
        self.viewer = napari.Viewer(title=f"QC Back-Projection — {self.group_id}")
        qtv = self.viewer.window.qt_viewer
        getattr(qtv, "_dockLayerList", qtv.dockLayerList).setVisible(False)
        getattr(qtv, "_dockLayerControls", qtv.dockLayerControls).setVisible(False)
        rows, cols = self._grid_layout(len(self.cam_codes))
        self.pt_layers: dict[str, napari.layers.Points] = {}
        self.sk_layers: dict[str, napari.layers.Shapes] = {}

        def _add_video_layer(vr, name, trans_xy):
            # Image layer is 3-D (t,y,x) when rgb=True, so translate needs (0, y, x)
            shp = getattr(vr, "shape", None)
            if shp is not None and len(shp) == 4 and shp[-1] in (3, 4):
                self.viewer.add_image(vr, name=name, rgb=True, blending="additive", translate=(0, *trans_xy))
            else:
                self.viewer.add_image(vr, name=name, blending="additive", translate=(0, *trans_xy))

        for idx, code in enumerate(self.cam_codes):
            vr = self.readers[code]
            r, c = divmod(idx, cols)
            trans_xy = (r * vr.shape[1], c * vr.shape[2])  # (y, x)
            label = self.view_code_to_name.get(code, code)

            _add_video_layer(vr, label, trans_xy)

            self.pt_layers[code] = self.viewer.add_points(
                data=np.zeros((0, 2), float),   # (y, x)
                size=12,
                name=f"{code}_kpts",
                face_color="white",             # replaced per-frame
                translate=trans_xy,             # (y, x)
            )

            self.sk_layers[code] = self.viewer.add_shapes(
                data=[],
                shape_type="path",
                edge_color="white",
                edge_width=2.0,
                name=f"{code}_skel",
                translate=trans_xy,             # (y, x)
            )

            # tolerant camera aliasing
            if code not in self.calib_map:
                print(f"[warn] Camera '{code}' not found in calibration; trying relaxed matching.")
                found = None
                for k in self.calib_map.keys():
                    if k == code or k.endswith(code) or code.endswith(k):
                        found = k; break
                if found is not None and found != code:
                    self.calib_map[code] = self.calib_map[found]
                elif code not in self.calib_map:
                    print(f"[warn] No calibration for '{code}'. Its overlay will remain empty.")

        # Caches & sync
        self.cam_pts: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        self.bus = _FrameBus()
        self.viewer.dims.events.current_step.connect(self._on_napari_step)

        # First frame + refresh
        self._current_frame = 0
        self._update_reprojections(0)
        try:
            self.viewer.reset_view()
        except Exception:
            pass

    # ---------------------- PUBLIC EXTERNAL CONTROL API (NEW) -------------------

    def current_frame(self) -> int:
        return int(self._current_frame)

    def max_frames(self) -> int:
        return int(self.T)

    def set_frame(self, fr: int) -> None:
        fr = int(np.clip(fr, 0, max(1, self.T) - 1))
        if getattr(self, "_current_frame", None) == fr:
            return
        try:
            steps = list(self.viewer.dims.current_step)
            steps[0] = fr
            self.viewer.dims.current_step = tuple(steps)
        except Exception:
            self._current_frame = fr
            self._update_reprojections(fr)
            self.bus.frameChanged.emit(fr)

    # ----------------------------- helpers ------------------------------------

    def _video_props(self, p: Path) -> Tuple[int, int, float, int]:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"Cannot open video {p}")
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps= float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return w, h, fps, n

    def _grid_layout(self, n_cam: int) -> Tuple[int, int]:
        cols = math.ceil(math.sqrt(n_cam))
        rows = math.ceil(n_cam / cols)
        return rows, cols

    # ----------------------------- Sync ---------------------------------------

    def _on_napari_step(self, event=None):
        fr = int(self.viewer.dims.current_step[0] if self.viewer.dims.ndim > 0 else 0)
        if fr != self._current_frame:
            self._current_frame = fr
            self._update_reprojections(fr)
            if hasattr(self, "bus"):
                self.bus.frameChanged.emit(fr)

    # -------------------------- Reprojection & update --------------------------

    def _update_reprojections(self, fr: int):
        # Build current world points & label list
        Xh    = self.X_head_per_frame.get(fr, np.zeros((0, 3), float))
        names = self.names_per_frame.get(fr, [])
        xform = self.xform_per_frame.get(fr, None)
        Xw    = _apply_head2world_if_present(Xh, xform)

        # Points per camera (cache projections per frame/cam)
        if Xw.size:
            for cam in self.pt_layers.keys():
                if fr not in self.cam_pts.get(cam, {}):
                    prm = self.calib_map.get(cam)
                    if prm is None:
                        self.cam_pts.setdefault(cam, {})[fr] = np.zeros((0, 2), float)
                        continue
                    try:
                        self.cam_pts.setdefault(cam, {})[fr] = _project_pts(Xw, prm)
                    except Exception:
                        self.cam_pts.setdefault(cam, {})[fr] = np.zeros((0, 2), float)

        # Apply unwanted-name filter for this frame
        def _filter_names_points(curr_names: List[str], uv: np.ndarray) -> tuple[List[str], np.ndarray]:
            if uv.size == 0 or not curr_names:
                return [], np.zeros((0, 2), float)
            keep_mask = np.array([not _is_unwanted(n) for n in curr_names], dtype=bool)
            return [n for n, k in zip(curr_names, keep_mask) if k], uv[keep_mask]

        # For skeleton segments we’ll need name->idx mapping after filtering
        for cam in self.pt_layers.keys():
            uv_full = self.cam_pts.get(cam, {}).get(fr, np.zeros((0, 2), float))
            names_filt, uv = _filter_names_points(names, uv_full)

            # --- Points layer (y, x) with rig_view-like colors ---
            if uv.size == 0:
                self.pt_layers[cam].data = np.zeros((0, 2), float)
                self.pt_layers[cam].properties = {}
            else:
                pts_yx = uv[:, [1, 0]]                    # (v,u) -> (y,x)
                self.pt_layers[cam].data = pts_yx
                labels = np.asarray(names_filt, dtype=object)
                self.pt_layers[cam].properties = {"label": labels}
                colors = np.array([self.name2color.get(n, (1.0, 1.0, 0.0)) for n in names_filt], float)  # default yellow
                self.pt_layers[cam].face_color = colors

            # --- Skeleton layer (list of 2-point paths in (y,x)) ---
            segs: List[np.ndarray] = []
            if uv.size and self.skeleton_edges and names_filt:
                name2idx = {n: i for i, n in enumerate(names_filt)}
                for a, b in self.skeleton_edges:
                    ia = name2idx.get(a); ib = name2idx.get(b)
                    if ia is None or ib is None:
                        continue
                    pa, pb = uv[ia], uv[ib]
                    if np.any(np.isnan(pa)) or np.isnan(pb).any():
                        continue
                    segs.append(np.array([[pa[1], pa[0]], [pb[1], pb[0]]], float))
            self.sk_layers[cam].data = segs

    def run(self):
        napari.run()
