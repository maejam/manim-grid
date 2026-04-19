import manim as m
import numpy as np
import pytest

from manim_grid.exceptions import GridValueError
from manim_grid.grid import EmptyMobject, Grid
from manim_grid.tags import Tags, TagsList


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
@pytest.fixture
def mobjects():
    return [
        m.Circle(color=m.RED),
        m.Square(color=m.BLUE),
        m.Dot(color=m.GREEN),
        m.Circle(color=m.BLUE),
        m.Square(color=m.GREEN),
        m.Dot(color=m.RED),
    ]


# ----------------------------------------------------------------------
# _BaseProxy
# ----------------------------------------------------------------------
def test_repr(simple_grid):
    assert repr(simple_grid.mobs) == "<MobsProxy of size (2, 3)>"
    assert repr(simple_grid.olds) == "<OldsProxy of size (2, 3)>"


def test_str(simple_grid):
    simple_grid.mobs[0, 0] = m.Square()
    simple_grid.mobs[0, 0] = m.Circle()
    assert (
        str(simple_grid.mobs) == "[['Circle' 'EmptyMobject' 'EmptyMobject']\n "
        "['EmptyMobject' 'EmptyMobject' 'EmptyMobject']]"
    )
    assert (
        str(simple_grid.olds) == "[['Square' 'EmptyMobject' 'EmptyMobject']\n "
        "['EmptyMobject' 'EmptyMobject' 'EmptyMobject']]"
    )


def test_iter(simple_grid):
    simple_grid.mobs[0, :] = [m.Square(), m.Circle(), m.Dot()]
    it = iter(simple_grid.mobs)
    assert isinstance(next(it), m.Square)
    assert isinstance(next(it), m.Circle)
    assert isinstance(next(it), m.Dot)
    assert isinstance(next(it), EmptyMobject)


def test_mask_with_keyword_filter(simple_grid, mobjects):
    simple_grid.mobs[:] = mobjects
    mask = simple_grid.mobs.mask(color=m.RED)
    expected = np.array([[True, False, False], [False, False, True]])
    np.testing.assert_array_equal(mask, expected)


def test_mask_with_multiple_keywords(simple_grid, mobjects):
    simple_grid.mobs[:] = mobjects
    mask = simple_grid.mobs.mask(color=m.RED, nonexistent_attr=123)
    expected = np.full((2, 3), False)
    np.testing.assert_array_equal(mask, expected)


def test_mask_with_predicate(simple_grid, mobjects):
    simple_grid.mobs[:] = mobjects
    mask = simple_grid.mobs.mask(predicate=lambda obj: isinstance(obj, m.Square))
    expected = np.array([[False, True, False], [False, True, False]])
    np.testing.assert_array_equal(mask, expected)


def test_mask_combines_predicate_and_keywords(simple_grid, mobjects):
    simple_grid.mobs[:] = mobjects
    mask = simple_grid.mobs.mask(
        predicate=lambda obj: isinstance(obj, m.Dot), color=m.RED
    )
    expected = np.array([[False, False, False], [False, False, True]])
    np.testing.assert_array_equal(mask, expected)


def test_mask_raises_when_no_filter_given(simple_grid):
    with pytest.raises(
        ValueError, match="You must provide a predicate or at least one keyword filter"
    ):
        simple_grid.mobs.mask()


# ----------------------------------------------------------------------
# RectsProxy
# ----------------------------------------------------------------------
def test_rects_proxy_is_readonly(simple_grid):
    with pytest.raises(
        TypeError, match="'RectsProxy' object does not support item assignment"
    ):
        simple_grid.rects[0, 0] = m.Rectangle()


# ----------------------------------------------------------------------
# MobsProxy
# ----------------------------------------------------------------------
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
        GridValueError, match="Bulk assignment requires a sequence of Mobjects."
    ):
        simple_grid.mobs[:, 0] = m.Square()


def test_error_when_assigning_with_non_matching_sequence_len(simple_grid):
    with pytest.raises(
        GridValueError, match="Length mismatch between the selected cells"
    ):
        simple_grid.mobs[0, :] = [m.Circle()]


# ----------------------------------------------------------------------
# OldsProxy
# ----------------------------------------------------------------------
def test_old_value_is_preserved_after_update(simple_grid):
    c = m.Circle()
    s = m.Square()

    simple_grid.mobs[0, 0] = c
    assert simple_grid.mobs[0, 0] is c
    assert simple_grid.olds[0, 0] is not c

    simple_grid.mobs[0, 0] = s
    assert simple_grid.mobs[0, 0] is s
    assert simple_grid.olds[0, 0] is c


def test_bulk_old_values_after_multiple_updates(simple_grid):
    objs = simple_grid.mobs[0, :]

    # New objects for the first row
    new_row = [m.Circle(), m.Triangle(), m.Square()]
    simple_grid.mobs[0, :] = new_row
    assert list(simple_grid.mobs[0, :]) == list(new_row)
    assert list(simple_grid.olds[0, :]) == list(objs)

    newer_row = [m.Dot(), m.Line(), m.Dot()]
    simple_grid.mobs[0, :] = newer_row
    assert list(simple_grid.mobs[0, :]) == list(newer_row)
    assert list(simple_grid.olds[0, :]) == list(new_row)


def test_olds_proxy_is_readonly(simple_grid):
    with pytest.raises(
        TypeError, match="'OldsProxy' object does not support item assignment"
    ):
        simple_grid.olds[0, 0] = m.Mobject()


# ----------------------------------------------------------------------
# TagsProxy
# ----------------------------------------------------------------------
def test_tagsproxy_getitem(simple_grid: Grid):
    assert simple_grid.tags[0, 0] == {}
    assert isinstance(simple_grid.tags[0, 0], Tags)
    simple_grid.tags[:].update({"baz": 42})
    assert simple_grid.tags[0, :] == [{"baz": 42}, {"baz": 42}, {"baz": 42}]
    assert isinstance(simple_grid.tags[0, :], TagsList)
    assert simple_grid.tags[0, 0] is not simple_grid.tags[0, 1]
