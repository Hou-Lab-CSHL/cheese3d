"""Regression checks for QC reprojection coordinate transformations."""

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from cheese3d_annotator.data_visualizer.qc_video import (
    _CachedPreviewReader,
    _MosaicPreviewReader,
    QCReprojApp,
    _apply_head2world_if_present,
)


def test_head_to_world_uses_anipose_inverse_for_nonorthogonal_matrix():
    """The saved Anipose axes matrix must not be treated as a pure rotation."""
    world = np.array([[2.0, -1.0, 4.0], [-3.0, 5.0, 0.5]])
    # Anipose may produce normalized axes that are not mutually orthogonal.
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.1, np.sqrt(0.99), 0.0],
        [0.0, 0.0, 1.0],
    ])
    center = np.array([0.5, -0.25, 1.25])
    head = world.dot(matrix.T) - center

    restored = _apply_head2world_if_present(head, (matrix, center))

    np.testing.assert_allclose(restored, world)


def test_cached_preview_reader_downscales_and_reuses_decoded_frame():
    """Interactive previews should cache resized frames instead of seeking again."""
    class FakeReader:
        shape = (4, 8, 10, 3)
        dtype = np.dtype("uint8")

        def __init__(self):
            self.calls = []

        def __getitem__(self, index):
            self.calls.append(index)
            return np.full((8, 10, 3), index, dtype=np.uint8)

    source = FakeReader()
    with ThreadPoolExecutor(max_workers=2) as executor:
        reader = _CachedPreviewReader(source, executor, scale=0.5, cache_size=2)
        first = reader[1]
        second = reader[1]

    assert reader.shape == (4, 4, 5, 3)
    assert first.shape == (4, 5, 3)
    np.testing.assert_array_equal(first, second)
    assert source.calls == [1]


def test_mosaic_preview_reader_combines_parallel_camera_frames():
    """Six separate Napari textures are replaced by one correctly tiled frame."""
    class FakeCachedReader:
        shape = (3, 2, 3, 3)

        def __init__(self, value):
            self.value = value
            self.prefetched = []

        def __len__(self):
            return self.shape[0]

        def prefetch(self, indices):
            self.prefetched.extend(indices)

        def __getitem__(self, index):
            return np.full(self.shape[1:], self.value + index, dtype=np.uint8)

    readers = {"A": FakeCachedReader(10), "B": FakeCachedReader(20)}
    mosaic = _MosaicPreviewReader(readers, ["A", "B"], rows=1, columns=2)

    frame = mosaic[1]

    assert mosaic.shape == (3, 2, 6, 3)
    assert np.all(frame[:, :3] == 11)
    assert np.all(frame[:, 3:] == 21)
    assert readers["A"].prefetched == [1]
    assert readers["B"].prefetched == [1]


def test_reprojections_are_precomputed_as_compact_float32_arrays():
    """Frame navigation should index cached projections instead of calling OpenCV."""
    app = QCReprojApp.__new__(QCReprojApp)
    app.T = 2
    app.bases = ["nose"]
    app.cam_codes = ["L"]
    app.names_per_frame = {0: ["nose"], 1: ["nose"]}
    app.X_head_per_frame = {
        0: np.array([[0.0, 0.0, 1.0]]),
        1: np.array([[1.0, 0.0, 1.0]]),
    }
    app.xform_per_frame = {}
    app.calib_map = {"L": {
        "K": np.eye(3), "dist": np.zeros(0),
        "rvec": np.zeros(3), "tvec": np.zeros(3),
    }}

    result = app._precompute_reprojections()["L"]

    assert result.shape == (2, 1, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, 0], [[0.0, 0.0], [1.0, 0.0]])
