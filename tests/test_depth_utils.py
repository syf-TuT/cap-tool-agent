import numpy as np

from capx.utils.depth_utils import depth_to_rgb


def test_depth_to_rgb_uses_available_matplotlib_colormap_registry() -> None:
    depth = np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float32)

    rgb = depth_to_rgb(depth, cmap_name="viridis")

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[1, 0], np.array([0, 0, 0], dtype=np.uint8))
    assert not np.array_equal(rgb[0, 0], rgb[1, 1])
