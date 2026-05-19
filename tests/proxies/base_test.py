import manim as m
import numpy as np
import pytest

from manim_grid.grid import EmptyMobject


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
        str(simple_grid.mobs) == "[['Mob(Circle)[0, 0]' 'Mob(EmptyMobject)[0, 1]' "
        "'Mob(EmptyMobject)[0, 2]']\n "
        "['Mob(EmptyMobject)[1, 0]' 'Mob(EmptyMobject)[1, 1]'\n  "
        "'Mob(EmptyMobject)[1, 2]']]"
    )
    assert (
        str(simple_grid.olds) == "[['Old(Square)[0, 0]' 'Old(EmptyMobject)[0, 1]' "
        "'Old(EmptyMobject)[0, 2]']\n "
        "['Old(EmptyMobject)[1, 0]' 'Old(EmptyMobject)[1, 1]'\n  "
        "'Old(EmptyMobject)[1, 2]']]"
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
