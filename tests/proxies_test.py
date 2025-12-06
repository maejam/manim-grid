import manim as m
import numpy as np
import pytest

from manim_grid.exceptions import GridValueError


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
    # default: m.ORIGIN.
    idx = (1, 1)
    simple_grid.mobs[idx] = dummy_mob
    assert dummy_mob.aligned_edge.all() == m.ORIGIN.all()

    # tuple alignment.
    tup_alignment = (0.2, 0.5, 0.0)
    idx = (1, 1, tup_alignment)
    simple_grid.mobs[idx] = dummy_mob
    assert dummy_mob.aligned_edge.all() == np.array(tup_alignment).all()

    # array alignment.
    arr_alignment = m.UP
    idx = (1, 0, arr_alignment)
    simple_grid.mobs[idx] = dummy_mob
    assert dummy_mob.aligned_edge.all() == arr_alignment.all()


def test_bulk_assignment_with_sequence(simple_grid):
    olds = simple_grid.mobs[:]
    objs = [m.Circle() for _ in range(6)]
    simple_grid.mobs[:] = objs

    assert simple_grid.mobs[:] == objs
    assert simple_grid.olds[:] == olds


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
    assert simple_grid.mobs[0, :] == new_row
    assert simple_grid.olds[0, :] == objs

    newer_row = [m.Dot(), m.Line(), m.Dot()]
    simple_grid.mobs[0, :] = newer_row
    assert simple_grid.mobs[0, :] == newer_row
    assert simple_grid.olds[0, :] == new_row


def test_olds_proxy_is_readonly(simple_grid):
    with pytest.raises(
        TypeError, match="'OldsProxy' object does not support item assignment"
    ):
        simple_grid.olds[0, 0] = m.Mobject()
