import manim as m
import numpy as np

from manim_grid import Grid


def test_alignment_proxy_getitem(simple_grid: Grid):
    assert np.array_equal(simple_grid.alignment[0, 0], m.ORIGIN)
    assert isinstance(simple_grid.alignment[0, 0], np.ndarray)
    simple_grid.alignment[0] = m.UP
    first_col = simple_grid.alignment[:, 0]
    assert isinstance(first_col, list)
    assert np.array_equal(first_col[0], m.UP)
    assert np.array_equal(first_col[1], m.ORIGIN)
    assert simple_grid.alignment[0, 0] is simple_grid.alignment[0, 1]
