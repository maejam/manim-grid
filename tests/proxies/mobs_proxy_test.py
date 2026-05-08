import manim as m
import numpy as np
import pytest

from manim_grid.exceptions import GridValueError


def test_mobs_proxy_basic_assignment_and_retrieval(simple_grid):
    circle = m.Circle(color=m.BLUE)
    simple_grid.mobs[0, 0] = circle

    retrieved = simple_grid.mobs[(0, 0)]
    assert retrieved is circle
    assert retrieved.get_color() == m.BLUE


def test_alignment_vector_is_respected(simple_grid, dummy_mob):
    # default default: m.ORIGIN.
    idx = (1, 1)
    simple_grid.mobs[idx] = dummy_mob
    assert np.array_equal(dummy_mob.aligned_edge, m.ORIGIN)

    # default: last alignment in cell.
    simple_grid.mobs[1, 1, m.UP] = dummy_mob
    assert np.array_equal(dummy_mob.aligned_edge, m.UP)
    simple_grid.mobs[1, 1] = dummy_mob
    assert np.array_equal(dummy_mob.aligned_edge, m.UP)

    # tuple alignment.
    tup_alignment = (0.2, 0.5, 0.0)
    idx = (1, 1, tup_alignment)
    simple_grid.mobs[idx] = dummy_mob
    assert np.array_equal(dummy_mob.aligned_edge, np.array(tup_alignment))

    # array alignment.
    arr_alignment = m.UP
    idx = (1, 0, arr_alignment)
    simple_grid.mobs[idx] = dummy_mob
    assert np.array_equal(dummy_mob.aligned_edge, arr_alignment)


def test_bulk_assignment_with_sequence(simple_grid):
    olds = simple_grid.mobs[:]
    objs = [m.Circle() for _ in range(6)]
    simple_grid.mobs[:] = objs

    assert isinstance(simple_grid.mobs[:], m.VGroup)
    assert isinstance(simple_grid.olds[:], m.VGroup)
    assert list(simple_grid.mobs[:]) == objs
    assert list(simple_grid.olds[:]) == list(olds)


def test_bulk_assignment_with_vgroup(simple_grid):
    olds = simple_grid.mobs[:]
    objs = m.VGroup([m.Circle() for _ in range(6)])
    simple_grid.mobs[:] = objs

    assert isinstance(simple_grid.mobs[:], m.VGroup)
    assert isinstance(simple_grid.olds[:], m.VGroup)
    assert list(simple_grid.mobs[:]) == list(objs)
    assert list(simple_grid.olds[:]) == list(olds)


def test_error_when_assigning_non_mobject(simple_grid):
    with pytest.raises(
        GridValueError, match="Only a single Mobject can be assigned to a single Cell"
    ):
        simple_grid.mobs[0, 0] = "not a mob"


def test_error_when_bulk_assignment_with_scalar_value(simple_grid):
    with pytest.raises(
        GridValueError,
        match=r"Bulk assignment requires a sequence or a \(V\)Group of Mobjects.",
    ):
        simple_grid.mobs[:, 0] = m.Square()


def test_error_when_assigning_with_non_matching_sequence_len(simple_grid):
    with pytest.raises(
        GridValueError, match="Length mismatch between the selected cells"
    ):
        simple_grid.mobs[0, :] = [m.Circle()]
