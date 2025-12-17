import manim as m
import pytest

from manim_grid import Grid


@pytest.fixture
def simple_grid():
    """A 2x2 grid."""
    row_heights = [1.0, 1.0]
    col_widths = [1.5, 1.5, 1.5]
    g = Grid(
        row_heights,
        col_widths,
        buff=(0.1, 0.3),
        margin=(0.1, 0.3),
        row_labels=(),
        col_labels=(),
    )
    return g


@pytest.fixture
def dummy_mob():
    class DummyMobject(m.Mobject):
        def __init__(self):
            self.pos = None
            self.aligned_edge = None
            self.shift_vec = None

        def move_to(self, target, aligned_edge=None):  # type:ignore[reportIncompatibleMethodOverride]
            self.pos = target
            self.aligned_edge = aligned_edge
            return self

        def shift(self, vec):  # type:ignore[reportIncompatibleMethodOverride]
            self.shift_vec = vec
            return self

        def __repr__(self):
            return f"<DummyMobject pos={self.pos} aligned={self.aligned_edge}>"

    return DummyMobject()
