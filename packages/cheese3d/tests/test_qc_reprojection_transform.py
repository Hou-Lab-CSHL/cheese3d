"""Regression checks for QC reprojection coordinate transformations."""

import numpy as np

from cheese3d_annotator.data_visualizer.qc_video import (
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
