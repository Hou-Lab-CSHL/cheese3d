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
import sys, re, math, json, warnings, threading, time, os
from pathlib import Path
from collections import defaultdict, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Tuple, List, Optional, Set
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


class _PyAVCudaReader:
    """Expose PyAV CUDA decoding through the NumPy-like interface Napari needs."""

    def __init__(self, filename: str, device: int = 0):
        import av
        from av.codec.hwaccel import HWAccel, hwdevices_available

        if "cuda" not in {str(item).lower() for item in hwdevices_available()}:
            raise RuntimeError("PyAV's loaded FFmpeg does not advertise CUDA decoding")
        self._av = av
        self._container = av.open(
            filename,
            hwaccel=HWAccel(
                "cuda", device=str(device), allow_software_fallback=False
            ),
        )
        self._stream = self._container.streams.video[0]
        self._fps = float(self._stream.average_rate or 30.0)
        self._start_time = int(self._stream.start_time or 0)
        self._length = int(self._stream.frames or 0)
        if self._length <= 0:
            raise RuntimeError(f"CUDA video stream does not report frame count: {filename}")
        self._iterator = iter(self._container.decode(self._stream))
        self._next_index = 0
        self.shape = (
            self._length, int(self._stream.height), int(self._stream.width), 3
        )
        self.dtype = np.dtype("uint8")

        # Force one real decode during initialization. Device/driver problems
        # therefore trigger the caller's CPU fallback before Napari opens.
        first = next(self._iterator)
        self._first_frame = first.to_ndarray(format="rgb24")
        self._next_index = 1

    def __len__(self) -> int:
        return self._length

    def _seek(self, index: int) -> None:
        """Seek to the preceding keyframe and resume decoding toward ``index``."""
        seconds = float(index) / self._fps
        timestamp = self._start_time + int(seconds / float(self._stream.time_base))
        self._container.seek(timestamp, stream=self._stream, backward=True, any_frame=False)
        self._iterator = iter(self._container.decode(self._stream))
        self._next_index = max(0, index - round(self._fps * 2))

    def __getitem__(self, index: int) -> np.ndarray:
        """Decode sequential frames cheaply and use timestamp seeking for scrubbing."""
        index = int(index)
        if index == 0 and self._first_frame is not None:
            return self._first_frame
        if index != self._next_index:
            self._seek(index)
        for frame in self._iterator:
            if frame.pts is not None:
                decoded_index = round(
                    (frame.pts - self._start_time) *
                    float(self._stream.time_base) * self._fps
                )
            else:
                decoded_index = self._next_index
            self._next_index = decoded_index + 1
            if decoded_index >= index:
                return frame.to_ndarray(format="rgb24")
        raise IndexError(f"Could not decode CUDA video frame {index}")


class _CachedPreviewReader:
    """Provide bounded, prefetched preview frames around a lazy video reader.

    Each camera owns a decode lock because its underlying reader may seek with
    mutable state, while the shared executor still decodes different cameras in
    parallel. Napari receives only NumPy frames and remains on the Qt thread.
    """

    def __init__(self, reader, executor: ThreadPoolExecutor,
                 scale: float = 0.5, cache_size: int = 8):
        self.reader = reader
        self.executor = executor
        self.scale = float(scale)
        self.cache_size = max(2, int(cache_size))
        raw_shape = tuple(reader.shape)
        self.shape = (
            raw_shape[0], max(1, round(raw_shape[1] * self.scale)),
            max(1, round(raw_shape[2] * self.scale)), *raw_shape[3:]
        )
        self.dtype = getattr(reader, "dtype", np.dtype("uint8"))
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._futures: Dict[int, Future] = {}
        self._lock = threading.Lock()
        self._decode_lock = threading.Lock()
        self._preload_stop = threading.Event()
        self._preload_thread: Optional[threading.Thread] = None
        self._interactive_until = 0.0

    def __len__(self) -> int:
        return int(self.shape[0])

    def min(self) -> int:
        """Expose the uint8 intensity range expected by Napari image layers."""
        return 0

    def max(self) -> int:
        """Expose the uint8 intensity range without decoding a complete video."""
        return 255

    def _decode(self, index: int) -> np.ndarray:
        """Decode one full frame and resize only the interactive preview copy."""
        with self._decode_lock:
            frame = np.asarray(self.reader[index])
        if self.scale != 1.0:
            frame = cv2.resize(
                frame, (self.shape[2], self.shape[1]), interpolation=cv2.INTER_AREA
            )
        return frame

    def prefetch(self, indices) -> None:
        """Schedule nearby frames without duplicating cached or active work."""
        with self._lock:
            for index in indices:
                index = int(index)
                if not 0 <= index < len(self) or index in self._cache \
                        or index in self._futures:
                    continue
                self._futures[index] = self.executor.submit(self._decode, index)

    def cancel_stale_prefetch(self, keep_indices) -> None:
        """Remove queued seeks made obsolete by rapid timeline scrubbing."""
        keep = {int(index) for index in keep_indices}
        with self._lock:
            for index, future in list(self._futures.items()):
                if index not in keep and (future.cancel() or future.done()):
                    # Dropping a completed stale future also releases its frame
                    # array instead of letting speculative results bypass LRU limits.
                    self._futures.pop(index, None)

    def _frame(self, index: int) -> np.ndarray:
        """Return one cached frame, waiting only for its already-prefetched future."""
        index = int(index)
        # Give foreground navigation exclusive decode priority for a short
        # window; preload workers check this timestamp between every frame.
        self._interactive_until = time.monotonic() + 0.4
        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached
        self.prefetch([index])
        with self._lock:
            future = self._futures.get(index)
        frame = future.result() if future is not None else self._decode(index)
        with self._lock:
            self._futures.pop(index, None)
        self._store_frame(index, frame)
        return frame

    def _store_frame(self, index: int, frame: np.ndarray) -> None:
        """Insert one decoded preview while enforcing the configured RAM limit."""
        with self._lock:
            self._cache[index] = frame
            self._cache.move_to_end(index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def start_background_preload(self, camera_name: str,
                                 preload_slots: threading.Semaphore) -> None:
        """Decode the timeline sequentially so later arbitrary seeks hit RAM."""
        if self._preload_thread is not None:
            return

        def preload() -> None:
            """Fill this camera's bounded cache without touching Napari widgets."""
            for index in range(len(self)):
                if self._preload_stop.is_set():
                    return
                while time.monotonic() < self._interactive_until:
                    if self._preload_stop.wait(timeout=0.02):
                        return
                with self._lock:
                    if index in self._cache:
                        continue
                # Only two cameras preload concurrently. Six simultaneous H.264
                # streams formerly saturated decode and delayed foreground seeks.
                with preload_slots:
                    self._store_frame(index, self._decode(index))
                if index and index % 5000 == 0:
                    print(f"QC preview preload {camera_name}: {index}/{len(self)} frames")
            print(f"QC preview preload {camera_name}: complete ({len(self)} frames)")

        self._preload_thread = threading.Thread(
            target=preload, name=f"cheese3d-preload-{camera_name}", daemon=True
        )
        self._preload_thread.start()

    def stop_background_preload(self) -> None:
        """Request cooperative shutdown when the visualizer closes."""
        self._preload_stop.set()

    def __getitem__(self, key):
        """Support Napari's scalar-time slicing while delegating unusual slices."""
        if isinstance(key, tuple) and key and isinstance(key[0], (int, np.integer)):
            frame = self._frame(int(key[0]))
            return frame[key[1:]] if len(key) > 1 else frame
        if isinstance(key, (int, np.integer)):
            return self._frame(int(key))
        # Preserve the underlying reader's behavior for metadata probes or
        # non-scalar time slices that Napari may issue during layer creation.
        return self.reader[key]


class _MosaicPreviewReader:
    """Expose all camera previews as one tiled lazy RGB video to Napari."""

    def __init__(self, readers: Dict[str, _CachedPreviewReader],
                 camera_codes: List[str], rows: int, columns: int):
        self.readers = readers
        self.camera_codes = camera_codes
        self.rows = int(rows)
        self.columns = int(columns)
        first = readers[camera_codes[0]]
        self.cell_height, self.cell_width = first.shape[1:3]
        self.shape = (
            min(len(reader) for reader in readers.values()),
            self.rows * self.cell_height, self.columns * self.cell_width, 3,
        )
        self.dtype = np.dtype("uint8")
        self.ndim = 4
        self.size = int(np.prod(self.shape))

    def __len__(self) -> int:
        return self.shape[0]

    def min(self) -> int:
        return 0

    def max(self) -> int:
        return 255

    def _frame(self, index: int) -> np.ndarray:
        """Wait for parallel camera decodes, then compose one GPU texture upload."""
        for reader in self.readers.values():
            reader.prefetch([index])
        canvas = np.zeros(self.shape[1:], dtype=np.uint8)
        for camera_index, code in enumerate(self.camera_codes):
            frame = self.readers[code][index]
            row, column = divmod(camera_index, self.columns)
            y0, x0 = row * self.cell_height, column * self.cell_width
            height = min(self.cell_height, frame.shape[0])
            width = min(self.cell_width, frame.shape[1])
            canvas[y0:y0 + height, x0:x0 + width] = frame[:height, :width, :3]
        return canvas

    def __getitem__(self, key):
        """Implement the scalar timeline slices requested by Napari."""
        if isinstance(key, tuple) and key and isinstance(key[0], (int, np.integer)):
            frame = self._frame(int(key[0]))
            return frame[key[1:]] if len(key) > 1 else frame
        if isinstance(key, (int, np.integer)):
            return self._frame(int(key))
        if isinstance(key, slice):
            return np.stack([self._frame(index) for index in range(*key.indices(len(self)))])
        raise TypeError(f"Unsupported mosaic video index: {key!r}")


class _PersistentMosaicReader:
    """Read a completed mosaic memory map or build it once in the background."""

    def __init__(self, source: _MosaicPreviewReader, cache_dir: Path,
                 source_paths: List[Path], preview_scale: float):
        self.source = source
        self.shape, self.dtype = source.shape, np.dtype("uint8")
        self.ndim, self.size = 4, int(np.prod(self.shape))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"camera-mosaic-{round(preview_scale * 1000):03d}.npy"
        self.cache_path = cache_dir / cache_name
        self.metadata_path = self.cache_path.with_suffix(".json")
        self._signature = {
            "shape": list(self.shape), "preview_scale": float(preview_scale),
            "sources": [{
                "path": str(path.resolve()), "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            } for path in source_paths],
        }
        self._mmap = None
        self._builder: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Disk previews are session-temporary. Clear files left by a crash or
        # forced shutdown before deciding whether a new map must be built.
        self._delete_cache_files()
        if self._is_valid():
            self._mmap = np.load(self.cache_path, mmap_mode="r")
            print(f"QC mosaic cache loaded: {self.cache_path}")
        else:
            self._start_builder()

    def _delete_cache_files(self) -> None:
        """Delete only Cheese3D mosaic artifacts within this session cache directory."""
        self._mmap = None
        candidates = [self.cache_path, self.metadata_path]
        candidates.extend(self.cache_path.parent.glob(f".{self.cache_path.name}.*.tmp"))
        for path in candidates:
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                print(f"QC mosaic cache cleanup warning for {path}: {error}")

    def _is_valid(self) -> bool:
        """Require both an atomic completion marker and matching source metadata."""
        try:
            return self.cache_path.is_file() and \
                json.loads(self.metadata_path.read_text()) == self._signature
        except (OSError, json.JSONDecodeError):
            return False

    def _start_builder(self) -> None:
        """Build a temporary map sequentially and publish it atomically when complete."""
        temporary = self.cache_path.with_name(
            f".{self.cache_path.name}.{os.getpid()}.tmp"
        )

        def build() -> None:
            """Decode the mosaic once while leaving the Qt event loop responsive."""
            try:
                mapped = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype=self.dtype, shape=self.shape
                )
                for index in range(len(self)):
                    if self._stop.is_set():
                        return
                    mapped[index] = self.source[index]
                    if index and index % 2500 == 0:
                        mapped.flush()
                        print(f"QC mosaic cache: {index}/{len(self)} frames")
                mapped.flush()
                del mapped
                temporary.replace(self.cache_path)
                metadata_tmp = self.metadata_path.with_suffix(".json.tmp")
                metadata_tmp.write_text(json.dumps(self._signature, sort_keys=True))
                metadata_tmp.replace(self.metadata_path)
                self._mmap = np.load(self.cache_path, mmap_mode="r")
                print(f"QC mosaic cache complete: {self.cache_path}")
            except Exception as error:
                print(f"QC mosaic cache build failed; continuing with live decode: {error}")
            finally:
                if temporary.exists():
                    temporary.unlink()

        self._builder = threading.Thread(
            target=build, name="cheese3d-mosaic-cache", daemon=True
        )
        self._builder.start()

    def __len__(self) -> int:
        return self.shape[0]

    def min(self) -> int:
        return 0

    def max(self) -> int:
        return 255

    def __getitem__(self, key):
        """Use zero-decode memory-map indexing once the persistent cache is ready."""
        mapped = self._mmap
        return mapped[key] if mapped is not None else self.source[key]

    def close(self) -> None:
        """Stop cache creation and remove all disk artifacts when Napari exits."""
        self._stop.set()
        if self._builder is not None:
            self._builder.join(timeout=2)
        self._delete_cache_files()
        try:
            self.cache_path.parent.rmdir()
        except OSError:
            # Keep the directory if it contains unrelated user files or a cache
            # builder is still unwinding; only known artifacts are ever deleted.
            pass

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


def _camera_code_from_pose_filename(path: Path, camera_codes: List[str]) -> Optional[str]:
    """Match an Anipose/DLC pose filename to one configured camera code.

    Camera codes are checked longest-first so ``TL`` is not accidentally
    interpreted as the one-letter camera ``L``.
    """
    stem = path.stem
    for code in sorted(camera_codes, key=len, reverse=True):
        # DLC scorer text may immediately follow the camera code, while the
        # character before it remains an underscore in Cheese3D filenames.
        if re.search(rf"_{re.escape(code)}(?=[A-Z_]|$)", stem):
            return code
    return None


def _load_pose2d_points(
    pose2d_dir: Optional[Path], camera_codes: List[str]
) -> Dict[str, object]:
    """Load DLC-compatible H5 detections as compact per-camera tables.

    Tables are retained instead of expanding every frame into nested dictionaries;
    this keeps long, multi-camera sessions from consuming gigabytes of Python
    object overhead. Unreadable or absent files are skipped so visualization of
    the 3D result remains available even when original detections were removed.
    """
    if pose2d_dir is None or not pose2d_dir.is_dir():
        return {}

    import pandas as pd

    result: Dict[str, object] = {}
    for h5_path in sorted(pose2d_dir.glob("*.h5")):
        cam = _camera_code_from_pose_filename(h5_path, camera_codes)
        if cam is None or cam in result:
            continue
        try:
            frame_table = pd.read_hdf(h5_path)
        except Exception as exc:
            print(f"[warn] Could not load original 2D poses from {h5_path}: {exc}")
            continue

        # Keep the former direct scorer selection visible for reference; the
        # generalized loop below also supports two-level and flat tables.
        # frame_table = frame_table.loc[:, frame_table.columns.levels[0][0]]
        if isinstance(frame_table.columns, pd.MultiIndex) and frame_table.columns.nlevels >= 3:
            frame_table = frame_table.loc[:, frame_table.columns.get_level_values(0)[0]]

        # The former eager expansion is intentionally not used because a full
        # session creates millions of small dictionaries and NumPy arrays:
        # cam_frames[fr][name] = np.array([x, y], dtype=float)
        result[cam] = frame_table
    return result


def _pose2d_points_for_frame(frame_table: object, fr: int) -> Dict[str, np.ndarray]:
    """Extract finite ``(x, y)`` coordinates for one frame from a pose table."""
    import pandas as pd

    if not isinstance(frame_table, pd.DataFrame) or not 0 <= fr < len(frame_table):
        return {}
    if not isinstance(frame_table.columns, pd.MultiIndex):
        return {}

    row = frame_table.iloc[fr]
    points: Dict[str, np.ndarray] = {}
    for name in frame_table.columns.get_level_values(0).unique():
        try:
            x, y = float(row[(name, "x")]), float(row[(name, "y")])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            points[str(name)] = np.array([x, y], dtype=float)
    return points

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
    Reverse Anipose's saved head-coordinate correction exactly.

    Anipose stores ``X_head = X_world.dot(M.T) - center`` and reconstructs
    world coordinates with ``(X_head + center).dot(inv(M.T))``. Although ``M``
    resembles a rotation matrix, Anipose normalizes its axes without fully
    orthogonalizing them; replacing the inverse with a transpose therefore
    creates a systematic one-sided reprojection shift.
    """
    if Xh.size == 0 or xform is None:
        return Xh
    M, center = xform
    # Former behavior assumed M was perfectly orthogonal. It is retained as
    # documentation because this shortcut caused the approximately 20 px shift:
    # return (M.T @ (Xh + center).T).T
    return (Xh + center).dot(np.linalg.inv(M.T))


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
    keypoint_views : Optional[Dict[str, List[str]]]
        Allowed long view names for each keypoint. Configured keypoints are hidden
        from cameras that did not contribute to their triangulation.
    pose2d_dir : Optional[Path | str]
        Directory containing original DLC-compatible H5 detections. When present,
        the viewer draws original points and residual vectors to reprojections.
    """
    def __init__(
        self,
        videos_by_group: Dict[str, Dict[str, str]],
        calibration_path: Path | str,
        pose3d_csv: Path | str,
        view_code_to_name: Optional[Dict[str, str]] = None,
        group: Optional[str] = None,
        skeleton_config: Optional[Path | List[Tuple[str, str]]] = None,
        keypoint_views: Optional[Dict[str, List[str]]] = None,
        pose2d_dir: Optional[Path | str] = None,
        preview_scale: float = 0.265,
        frame_cache_size: int = 8,
        cache_memory_gb: float = 32.0,
        slider_debounce_ms: int = 110,
        cuda_decode: bool = False,
        cuda_device: int = 0,
        playback_fps: float = 30.0,
        persistent_cache: bool = True,
        persistent_cache_disk_gb: float = 10.0,
        cache_dir: Optional[Path | str] = None,
        hide_overlays_during_playback: bool = False,
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
        if not 0 < preview_scale <= 1:
            raise ValueError("preview_scale must be greater than 0 and at most 1")
        if cache_memory_gb <= 0 or frame_cache_size <= 0 or slider_debounce_ms < 0:
            raise ValueError(
                "Visualization cache memory and frame cache must be positive; "
                "slider debounce must be non-negative"
            )
        self.preview_scale = float(preview_scale)
        self.hide_overlays_during_playback = bool(hide_overlays_during_playback)
        # An empty mapping intentionally preserves compatibility for callers that
        # have no per-keypoint view configuration and therefore must show all.
        self.keypoint_views: Dict[str, Set[str]] = {
            name: set(views) for name, views in (keypoint_views or {}).items()
        }

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
        # Original 2D detections are loaded once and indexed for fast frame seeks.
        self.pose2d_points = _load_pose2d_points(
            Path(pose2d_dir) if pose2d_dir is not None else None,
            self.cam_codes,
        )

        # Skeleton edges (optional)
        if isinstance(skeleton_config, list):
            self.skeleton_edges = skeleton_config
        else:
            self.skeleton_edges = _load_skeleton_edges(skeleton_config)

        # Open video readers; gather sizes & timeline length
        # Open one lazy reader per camera, then share a bounded executor so
        # independent camera seeks decode concurrently instead of serially.
        self._decode_executor = ThreadPoolExecutor(
            max_workers=max(1, min(len(self.cam_codes), 8)),
            thread_name_prefix="cheese3d-preview",
        )
        self._preload_slots = threading.Semaphore(2)
        self.readers: dict[str, _CachedPreviewReader] = {}
        counts = []
        ref_w = ref_h = None
        for code, vpath in zip(self.cam_codes, self.vids):
            if cuda_decode:
                try:
                    raw_reader = _PyAVCudaReader(str(vpath), device=int(cuda_device))
                    print(f"QC preview {code}: CUDA/NVDEC decoding on GPU {cuda_device}")
                except Exception as error:
                    print(
                        f"QC preview {code}: CUDA decode unavailable ({error}); "
                        "falling back to CPU"
                    )
                    raw_reader = VideoReaderNP(str(vpath))
            else:
                raw_reader = VideoReaderNP(str(vpath))
            raw_shape = tuple(raw_reader.shape)
            preview_bytes = max(
                1, round(raw_shape[1] * self.preview_scale) *
                round(raw_shape[2] * self.preview_scale) * 3
            )
            # Use at most 75% of the allowed budget for cached arrays, leaving
            # headroom for Napari textures, overlays, tables, and Python itself.
            budget_frames = int(
                max(0.25, float(cache_memory_gb)) * (1024 ** 3) * 0.75 /
                (max(1, len(self.cam_codes)) * preview_bytes)
            )
            camera_cache_size = max(
                int(frame_cache_size), min(int(raw_shape[0]), budget_frames)
            )
            rdr = _CachedPreviewReader(
                raw_reader, self._decode_executor,
                scale=self.preview_scale, cache_size=camera_cache_size,
            )
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
            raw_w, raw_h, _, _ = self._video_props(self.vids[0])
            ref_w = max(1, round(raw_w * self.preview_scale))
            ref_h = max(1, round(raw_h * self.preview_scale))
        self.video_size = (ref_w, ref_h)
        self.T = int(min(counts)) if counts else 1
        print(
            f"QC preview: scale={self.preview_scale:g}, cache budget={cache_memory_gb:g} GiB, "
            f"up to {camera_cache_size} frames per camera"
        )

        # Calibration (scaled to video size) with tolerant camera naming
        raw_cal = _load_calibration_raw(self.calib_path)
        self.calib_map = _build_calib_map(raw_cal, self.video_size)   # keyed by short cam code ('BC','L',...)
        self.projected_points = self._precompute_reprojections()

        # Napari viewer & layers
        self.viewer = napari.Viewer(title=f"QC Back-Projection — {self.group_id}")
        # Collapse rapid slider events into one displayed frame. Without this,
        # dragging across 100 positions launches 600 camera seeks that are
        # obsolete before decoding completes.
        self._pending_frame = 0
        self._frame_timer = QtCore.QTimer()
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(int(slider_debounce_ms))
        self._frame_timer.timeout.connect(self._commit_pending_frame)
        self._playback_timer = QtCore.QTimer()
        self._playback_timer.timeout.connect(self._advance_playback_frame)
        qtv = self.viewer.window.qt_viewer
        getattr(qtv, "_dockLayerList", qtv.dockLayerList).setVisible(False)
        getattr(qtv, "_dockLayerControls", qtv.dockLayerControls).setVisible(False)
        self._add_playback_controls(playback_fps)
        rows, cols = self._grid_layout(len(self.cam_codes))
        self.pt_layers: dict[str, napari.layers.Points] = {}
        self.original_pt_layers: dict[str, napari.layers.Points] = {}
        self.sk_layers: dict[str, napari.layers.Shapes] = {}
        self.residual_layers: dict[str, napari.layers.Shapes] = {}
        self.camera_translations: Dict[str, np.ndarray] = {}

        def _add_video_layer(vr, name, trans_xy):
            # Image layer is 3-D (t,y,x) when rgb=True, so translate needs (0, y, x)
            shp = getattr(vr, "shape", None)
            if shp is not None and len(shp) == 4 and shp[-1] in (3, 4):
                self.viewer.add_image(vr, name=name, rgb=True, blending="additive", translate=(0, *trans_xy))
            else:
                self.viewer.add_image(vr, name=name, blending="additive", translate=(0, *trans_xy))

        # Six independent image layers formerly caused six texture uploads and
        # redraws per slider position. One lazy mosaic retains camera layout but
        # updates Napari's canvas only once.
        mosaic_reader = _MosaicPreviewReader(
            self.readers, self.cam_codes, rows=rows, columns=cols
        )
        required_cache_bytes = int(np.prod(mosaic_reader.shape) * np.dtype("uint8").itemsize)
        disk_limit_bytes = int(float(persistent_cache_disk_gb) * (1024 ** 3))
        self.persistent_mosaic_reader = None
        if persistent_cache and required_cache_bytes <= disk_limit_bytes:
            self.persistent_mosaic_reader = _PersistentMosaicReader(
                mosaic_reader,
                Path(cache_dir) if cache_dir is not None else self.pose3d_csv.parent / "visualization-cache",
                self.vids, self.preview_scale,
            )
            mosaic_reader = self.persistent_mosaic_reader
        elif persistent_cache:
            print(
                f"QC persistent mosaic disabled: requires "
                f"{required_cache_bytes / (1024 ** 3):.2f} GiB, exceeds "
                f"{persistent_cache_disk_gb:g} GiB disk limit"
            )
        self.viewer.add_image(
            mosaic_reader, name="camera grid", rgb=True, blending="additive"
        )

        for idx, code in enumerate(self.cam_codes):
            vr = self.readers[code]
            r, c = divmod(idx, cols)
            trans_xy = (r * vr.shape[1], c * vr.shape[2])  # (y, x)
            self.camera_translations[code] = np.asarray(trans_xy, dtype=float)
            label = self.view_code_to_name.get(code, code)

            # Superseded by the single mosaic layer above:
            # _add_video_layer(vr, label, trans_xy)

            self.pt_layers[code] = self.viewer.add_points(
                data=np.zeros((0, 2), float),   # (y, x)
                size=4,
                name=f"{code}_kpts",
                face_color="white",             # replaced per-frame
                translate=trans_xy,             # (y, x)
            )

            # Original detections use hollow-looking cyan rings so they remain
            # visually distinct from the solid colored 3D reprojections.
            self.original_pt_layers[code] = self.viewer.add_points(
                data=np.zeros((0, 2), float),
                size=6,
                name=f"{code}_original_2d",
                face_color="transparent",
                border_color="cyan",
                border_width=0.15,
                translate=trans_xy,
            )

            self.sk_layers[code] = self.viewer.add_shapes(
                data=[],
                shape_type="path",
                edge_color="white",
                edge_width=1.25,
                name=f"{code}_skel",
                translate=trans_xy,             # (y, x)
            )


            # Each yellow path runs from the original 2D location to the 3D
            # reprojection; its length is the camera-specific residual in pixels.
            self.residual_layers[code] = self.viewer.add_shapes(
                data=[],
                shape_type="path",
                edge_color="yellow",
                edge_width=1.5,
                name=f"{code}_residuals",
                translate=trans_xy,
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

        # The per-camera layers above are retained for source compatibility but
        # superseded by four global mosaic layers to reduce 24 mutations/frame.
        for layer_map in (
            self.pt_layers, self.original_pt_layers,
            self.sk_layers, self.residual_layers,
        ):
            for layer in layer_map.values():
                layer.visible = False
        self.mosaic_pt_layer = self.viewer.add_points(
            np.zeros((0, 2), float), size=4, name="all_reprojected_keypoints",
            face_color="white",
        )
        self.mosaic_original_layer = self.viewer.add_points(
            np.zeros((0, 2), float), size=6, name="all_original_2d",
            face_color="transparent", border_color="cyan", border_width=0.15,
        )
        self.mosaic_skeleton_layer = self.viewer.add_shapes(
            data=[], shape_type="path", edge_color="white", edge_width=1.25,
            name="all_skeletons",
        )
        self.mosaic_residual_layer = self.viewer.add_shapes(
            data=[], shape_type="path", edge_color="yellow", edge_width=1.5,
            name="all_residuals",
        )

        # Caches & sync
        # Former per-frame projection dictionary is superseded by the compact
        # precomputed float32 arrays built before the viewer starts:
        # self.cam_pts: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        self.bus = _FrameBus()
        self.viewer.dims.events.current_step.connect(self._on_napari_step)

        # First frame + refresh
        self._current_frame = 0
        self._prefetch_video_frames(0)
        self._update_reprojections(0)
        for code, reader in self.readers.items():
            reader.start_background_preload(code, self._preload_slots)
        try:
            self.viewer.reset_view()
        except Exception:
            pass

    # ---------------------- PUBLIC EXTERNAL CONTROL API (NEW) -------------------

    def current_frame(self) -> int:
        # Expose a queued slider position so repeated keyboard steps accumulate
        # correctly during the short debounce window.
        if hasattr(self, "_frame_timer") and self._frame_timer.isActive():
            return int(self._pending_frame)
        return int(self._current_frame)

    def max_frames(self) -> int:
        return int(self.T)

    def set_frame(self, fr: int) -> None:
        fr = int(np.clip(fr, 0, max(1, self.T) - 1))
        if getattr(self, "_current_frame", None) == fr and not self._frame_timer.isActive():
            return
        self._pending_frame = fr
        self._frame_timer.start()

    def _commit_pending_frame(self) -> None:
        """Apply only the most recent externally requested slider position."""
        fr = int(self._pending_frame)
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

    def _add_playback_controls(self, initial_fps: float) -> None:
        """Add visible Play/Pause and frame-rate controls to the camera viewer."""
        controls = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(controls)
        layout.setContentsMargins(6, 3, 6, 3)
        self._play_button = QtWidgets.QPushButton("Play")
        self._play_button.setCheckable(True)
        self._fps_spinner = QtWidgets.QDoubleSpinBox()
        self._fps_spinner.setRange(0.5, 240.0)
        self._fps_spinner.setDecimals(1)
        self._fps_spinner.setSingleStep(5.0)
        self._fps_spinner.setSuffix(" FPS")
        self._fps_spinner.setValue(float(np.clip(initial_fps, 0.5, 240.0)))
        layout.addWidget(self._play_button)
        layout.addWidget(QtWidgets.QLabel("Playback rate:"))
        layout.addWidget(self._fps_spinner)
        layout.addStretch(1)
        self._play_button.toggled.connect(self._toggle_playback)
        self._fps_spinner.valueChanged.connect(self._update_playback_interval)
        self.viewer.window.add_dock_widget(
            controls, area="bottom", name="Camera playback"
        )
        self._update_playback_interval(self._fps_spinner.value())

    def _update_playback_interval(self, fps: float) -> None:
        """Apply a changed GUI frame rate without restarting visualization."""
        self._playback_timer.setInterval(max(1, round(1000.0 / float(fps))))

    def _toggle_playback(self, playing: bool) -> None:
        """Start or stop camera playback from the physical GUI button."""
        self._play_button.setText("Pause" if playing else "Play")
        if playing:
            self._playback_anchor_time = time.monotonic()
            self._playback_anchor_frame = self.current_frame()
            self._playback_timer.start()
            if self.hide_overlays_during_playback:
                for layer in (
                    self.mosaic_pt_layer, self.mosaic_original_layer,
                    self.mosaic_skeleton_layer, self.mosaic_residual_layer,
                ):
                    layer.visible = False
        else:
            self._playback_timer.stop()
            if self.hide_overlays_during_playback:
                for layer in (
                    self.mosaic_pt_layer, self.mosaic_original_layer,
                    self.mosaic_skeleton_layer, self.mosaic_residual_layer,
                ):
                    layer.visible = True
                self._update_reprojections(self.current_frame())

    def _advance_playback_frame(self) -> None:
        """Follow source time, dropping frames when rendering cannot reach target FPS."""
        elapsed = time.monotonic() - self._playback_anchor_time
        elapsed_frames = int(elapsed * float(self._fps_spinner.value()))
        next_frame = (self._playback_anchor_frame + elapsed_frames) % max(1, self.T)
        if next_frame == self.current_frame():
            return
        self._pending_frame = next_frame
        self._frame_timer.stop()
        self._commit_pending_frame()

    # ----------------------------- Sync ---------------------------------------

    def _on_napari_step(self, event=None):
        fr = int(self.viewer.dims.current_step[0] if self.viewer.dims.ndim > 0 else 0)
        if fr != self._current_frame:
            self._current_frame = fr
            self._prefetch_video_frames(fr)
            self._update_reprojections(fr)
            if hasattr(self, "bus"):
                self.bus.frameChanged.emit(fr)

    # -------------------------- Reprojection & update --------------------------

    def _precompute_reprojections(self) -> Dict[str, np.ndarray]:
        """Batch all 3D-to-2D projections into compact camera arrays once."""
        base_index = {name: index for index, name in enumerate(self.bases)}
        world = np.full((self.T, len(self.bases), 3), np.nan, dtype=np.float64)
        for frame in range(self.T):
            names = self.names_per_frame.get(frame, [])
            points = self.X_head_per_frame.get(frame, np.zeros((0, 3), float))
            transformed = _apply_head2world_if_present(
                points, self.xform_per_frame.get(frame)
            )
            for name, point in zip(names, transformed):
                index = base_index.get(name)
                if index is not None:
                    world[frame, index] = point

        finite = np.isfinite(world).all(axis=2)
        flattened = world.reshape(-1, 3)
        flattened_finite = finite.ravel()
        projected: Dict[str, np.ndarray] = {}
        for camera in self.cam_codes:
            camera_points = np.full(
                (self.T * len(self.bases), 2), np.nan, dtype=np.float32
            )
            parameters = self.calib_map.get(camera)
            if parameters is not None and flattened_finite.any():
                camera_points[flattened_finite] = _project_pts(
                    flattened[flattened_finite], parameters
                ).astype(np.float32)
            projected[camera] = camera_points.reshape(self.T, len(self.bases), 2)
        print(
            f"QC reprojections precomputed: {self.T} frames × "
            f"{len(self.bases)} keypoints × {len(projected)} cameras"
        )
        return projected

    def _prefetch_video_frames(self, fr: int) -> None:
        """Decode the current and adjacent frames across all cameras concurrently."""
        nearby = [index for index in (fr, fr + 1, fr - 1, fr + 2)
                  if 0 <= index < self.T]
        for reader in self.readers.values():
            reader.cancel_stale_prefetch(nearby)
        # Offset-first scheduling puts every camera's requested frame ahead of
        # speculative neighbors in the shared executor queue.
        for index in nearby:
            for reader in self.readers.values():
                reader.prefetch([index])

    def _update_reprojections(self, fr: int):
        """Update four consolidated mosaic overlays for one preprojected frame."""
        names = self.names_per_frame.get(fr, [])
        name_indices = [self.bases.index(name) for name in names if name in self.bases]

        # Apply unwanted-name filter for this frame
        def _filter_names_points(curr_names: List[str], uv: np.ndarray) -> tuple[List[str], np.ndarray]:
            if uv.size == 0 or not curr_names:
                return [], np.zeros((0, 2), float)
            keep_mask = np.array([not _is_unwanted(n) for n in curr_names], dtype=bool)
            return [n for n, k in zip(curr_names, keep_mask) if k], uv[keep_mask]

        def _filter_for_camera(
            cam: str, curr_names: List[str], uv: np.ndarray
        ) -> tuple[List[str], np.ndarray]:
            """Keep only keypoints configured for this camera's named view."""
            if uv.size == 0 or not curr_names:
                return [], np.zeros((0, 2), float)
            if not self.keypoint_views:
                return curr_names, uv
            view_name = self.view_code_to_name.get(cam, cam)
            keep_mask = np.array([
                view_name in self.keypoint_views.get(name, {view_name})
                for name in curr_names
            ], dtype=bool)
            return [name for name, keep in zip(curr_names, keep_mask) if keep], uv[keep_mask]

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        all_labels: List[str] = []
        all_original: List[np.ndarray] = []
        all_residuals: List[np.ndarray] = []
        all_skeletons: List[np.ndarray] = []

        # Accumulate camera coordinates in mosaic space, then mutate each
        # Napari overlay exactly once after the loop.
        for cam in self.cam_codes:
            translation = self.camera_translations[cam]
            camera_points = self.projected_points.get(cam)
            uv_full = camera_points[fr, name_indices] if camera_points is not None \
                else np.zeros((0, 2), float)
            names_filt, uv = _filter_names_points(names, uv_full)
            # Former behavior displayed every reconstructed keypoint in every
            # camera. Keep it documented because it explained misleading offsets:
            # names_filt, uv = _filter_names_points(names, uv_full)
            names_filt, uv = _filter_for_camera(cam, names_filt, uv)

            if uv.size:
                all_points.extend(uv[:, [1, 0]] + translation)
                all_labels.extend(names_filt)
                all_colors.extend(
                    self.name2color.get(name, (1.0, 1.0, 0.0))
                    for name in names_filt
                )

            # Show original detections and connect them to matching reprojections.
            # Only materialize detections for the displayed frame to keep memory
            # bounded for recordings containing tens of thousands of frames.
            original_by_name = _pose2d_points_for_frame(
                self.pose2d_points.get(cam), fr
            )
            if self.preview_scale != 1.0:
                original_by_name = {
                    name: point * self.preview_scale
                    for name, point in original_by_name.items()
                }
            original_yx: List[np.ndarray] = []
            residuals: List[np.ndarray] = []
            uv_by_name = {name: point for name, point in zip(names_filt, uv)}
            for name in names_filt:
                original = original_by_name.get(name)
                projected = uv_by_name.get(name)
                if original is None or projected is None:
                    continue
                original_yx.append(original[[1, 0]] + translation)
                residuals.append(np.array([
                    original[[1, 0]] + translation,
                    projected[[1, 0]] + translation,
                ], dtype=float))
            all_original.extend(original_yx)
            all_residuals.extend(residuals)

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
                    segs.append(
                        np.array([[pa[1], pa[0]], [pb[1], pb[0]]], float)
                        + translation
                    )
            all_skeletons.extend(segs)

        self.mosaic_pt_layer.data = np.asarray(all_points, dtype=float).reshape(-1, 2)
        self.mosaic_pt_layer.properties = {
            "label": np.asarray(all_labels, dtype=object)
        }
        if all_colors:
            self.mosaic_pt_layer.face_color = np.asarray(all_colors, dtype=float)
        self.mosaic_original_layer.data = np.asarray(
            all_original, dtype=float
        ).reshape(-1, 2)
        self.mosaic_residual_layer.data = all_residuals
        self.mosaic_skeleton_layer.data = all_skeletons

    def run(self):
        try:
            napari.run()
        finally:
            # Preview workers own no GUI objects and can be cancelled once the
            # Napari event loop closes.
            for reader in self.readers.values():
                reader.stop_background_preload()
            self._playback_timer.stop()
            if self.persistent_mosaic_reader is not None:
                self.persistent_mosaic_reader.close()
            self._decode_executor.shutdown(wait=False, cancel_futures=True)
