import manim as m
import pytest


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
