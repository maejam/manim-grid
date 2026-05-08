import manim as m
import pytest


def test_rects_proxy_is_readonly(simple_grid):
    with pytest.raises(
        TypeError, match="'RectsProxy' object does not support item assignment"
    ):
        simple_grid.rects[0, 0] = m.Rectangle()
