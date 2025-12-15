import manim as m
import numpy as np
import pytest

from manim_grid.exceptions import GridValueError
from manim_grid.grid import EmptyMobject, Grid
from manim_grid.proxies.tags_proxy import MISSING, Tags


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

    assert isinstance(simple_grid.mobs[:], m.VGroup)
    assert isinstance(simple_grid.olds[:], m.VGroup)
    assert list(simple_grid.mobs[:]) == objs
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
def test_tags_str():
    tags = Tags(one=1, two=2)
    assert str(tags) == "{'one': 1, 'two': 2}"


def test_tags_repr():
    tags = Tags(one=1, two=2)
    assert repr(tags) == "Tags(one=1, two=2)"


def test_scalar_tags_selection_setattr_getattr_and_delattr(simple_grid: Grid):
    simple_grid.tags[1, 1].foo = "bar"
    assert simple_grid.tags[1, 1].foo == "bar"
    del simple_grid.tags[1, 1].foo
    assert simple_grid.tags[1, 1].foo is MISSING
    assert simple_grid.tags[1, 1].baz is MISSING
    assert simple_grid.tags[0, 0].foo is MISSING
    with pytest.raises(AttributeError):
        del simple_grid.tags[1, 1].foo

    simple_grid.tags[0, 0] = {"foo": "bar", "baz": 42}
    assert simple_grid.tags[0, 0].foo == "bar"
    assert simple_grid.tags[0, 0].baz == 42


def test_scalar_tags_selection_update(simple_grid: Grid):
    simple_grid.tags[1, 1].update(foo="bar", baz=42)
    assert simple_grid.tags[1, 1].foo == "bar"
    assert simple_grid.tags[1, 1].baz == 42
    assert simple_grid.tags[0, 0].foo is MISSING
    assert simple_grid.tags[1, 1].foobar is MISSING


def test_scalar_tags_selection_remove(simple_grid: Grid):
    simple_grid.tags[1, 1].update(foo="bar", baz=42)
    simple_grid.tags[1, 1].remove("baz")
    assert simple_grid.tags[1, 1].foo == "bar"
    assert simple_grid.tags[1, 1].baz is MISSING
    assert simple_grid.tags[0, 0].foo is MISSING
    assert simple_grid.tags[1, 1].foobar is MISSING


def test_scalar_tags_selection_clear(simple_grid: Grid):
    simple_grid.tags[1, 1].update(foo="bar", baz=42)
    simple_grid.tags[1, 1].clear()
    assert simple_grid.tags[1, 1].foo is MISSING
    assert simple_grid.tags[1, 1].baz is MISSING
    assert simple_grid.tags[0, 0].foo is MISSING
    assert simple_grid.tags[1, 1].foobar is MISSING


def test_scalar_tags_selection_str(simple_grid: Grid):
    assert str(simple_grid.tags[1, 1]) == "{}"
    simple_grid.tags[1, 1].foo = "bar"
    assert str(simple_grid.tags[1, 1]) == "{'foo': 'bar'}"
    assert str(simple_grid.tags[1, 1].dne) == "<MISSING>"


def test_scalar_tags_selection_repr(simple_grid: Grid):
    assert (
        repr(simple_grid.tags[1, 1])
        == "ScalarTagsSelection(cell=Cell(rect=Rectangle, mob=EmptyMobject, "
        "old=EmptyMobject, tags=Tags()))"
    )
    simple_grid.tags[1, 1].foo = "bar"
    assert (
        repr(simple_grid.tags[1, 1])
        == "ScalarTagsSelection(cell=Cell(rect=Rectangle, mob=EmptyMobject, "
        "old=EmptyMobject, tags=Tags(foo=bar)))"
    )


def test_bulk_tags_selection_setattr_getattr_and_delattr(simple_grid: Grid):
    simple_grid.tags[0, :].foo = "bar"
    it = iter(simple_grid.tags[0, :])
    assert next(it)["foo"] == "bar"
    assert next(it).foo == "bar"
    assert next(it).foo == "bar"
    with pytest.raises(StopIteration):
        _ = next(it).foo
    simple_grid.tags[1, :] = Tags(foo="bar", baz=42)
    del simple_grid.tags[1, :].foo
    assert not any(simple_grid.tags[1, :].foo == "bar")
    assert all(simple_grid.tags[1, :].baz == 42)


def test_setattr_raises_with_invalid_input(simple_grid: Grid):
    with pytest.raises(
        TypeError,
        match="TagsProxy expects a Tags instance or a mapping",
    ):
        simple_grid.tags[1, :] = 42


def test_bulk_tags_selection_update(simple_grid: Grid):
    simple_grid.tags[1, :].update(foo="bar", baz=42)
    for i in range(3):
        assert simple_grid.tags[1, i].foo == "bar"
        assert simple_grid.tags[1, i].baz == 42
        assert simple_grid.tags[1, i].foobar is MISSING
    assert simple_grid.tags[0, 0].foo is MISSING


def test_keys_are_validated_when_mutating_tags(simple_grid: Grid):
    with pytest.raises(ValueError, match="Tag keys may not start with '_'"):
        simple_grid.tags[1, 1]._foo = "bar"
    with pytest.raises(ValueError, match="Tag keys may not start with '_'"):
        simple_grid.tags[1, :]._foo = "bar"
    with pytest.raises(ValueError, match="is not a valid Python identifier"):
        simple_grid.tags[1, 1] = {"9foo": "bar"}
    with pytest.raises(ValueError, match="is not a valid Python identifier"):
        simple_grid.tags[1, :] = {"9foo": "bar"}
    with pytest.raises(ValueError, match="is not a valid Python identifier"):
        simple_grid.tags[1, :] = {"foo baz": "bar"}


def test_bulk_tags_selection_remove(simple_grid: Grid):
    simple_grid.tags[1, :].update(foo="bar", baz=42)
    simple_grid.tags[1, 0:2].remove("baz")
    assert simple_grid.tags[1, 0].foo == "bar"
    assert simple_grid.tags[1, 0].baz is MISSING
    assert simple_grid.tags[1, 1].foo == "bar"
    assert simple_grid.tags[1, 1].baz is MISSING
    assert simple_grid.tags[1, 2].foo == "bar"
    assert simple_grid.tags[1, 2].baz == 42
    assert simple_grid.tags[0, 0].foo is MISSING
    assert simple_grid.tags[1, 1].foobar is MISSING


def test_bulk_tags_selection_clear(simple_grid: Grid):
    simple_grid.tags[1, :].update(foo="bar", baz=42)
    simple_grid.tags[1, 1:].clear()
    assert simple_grid.tags[1, 0].foo == "bar"
    assert simple_grid.tags[1, 0].baz == 42
    assert simple_grid.tags[1, 1].foo is MISSING
    assert simple_grid.tags[1, 1].baz is MISSING
    assert simple_grid.tags[0, 0].foo is MISSING
    assert simple_grid.tags[1, 1].foobar is MISSING


def test_tags_proxy_mask(simple_grid: Grid):
    mob1 = m.Circle()
    mob2 = m.Square()
    mob3 = m.Dot()
    mob4 = m.Rectangle()
    mob5 = m.Rectangle()
    mob6 = m.Rectangle()
    simple_grid.mobs[0] = [mob1, mob2, mob3]
    simple_grid.mobs[1] = [mob4, mob5, mob6]
    simple_grid.tags[0] = {"foo": "bar", "baz": 42}
    mask = simple_grid.tags.mask(predicate=lambda d: "foo" in d, baz=42)
    assert list(simple_grid.mobs[mask]) == [mob1, mob2, mob3]

    mask2 = simple_grid.tags.mask(baz=MISSING)
    assert list(simple_grid.mobs[mask2]) == [mob4, mob5, mob6]


def test_bulk_tags_selection_str(simple_grid: Grid):
    assert str(simple_grid.tags[1, :]) == "['{}' '{}' '{}']"
    simple_grid.tags[1, :].foo = "bar"
    assert (
        str(simple_grid.tags[1, :])
        == "[\"{'foo': 'bar'}\" \"{'foo': 'bar'}\" \"{'foo': 'bar'}\"]"
    )


def test_bulk_tags_selection_repr(simple_grid: Grid):
    assert (
        repr(simple_grid.tags[1, :1])
        == "BulkTagsSelection(cells=array([Cell(rect=Rectangle, mob=EmptyMobject, "
        "old=EmptyMobject, tags=Tags())],\n      dtype=object))"
    )
    simple_grid.tags[1, :].foo = "bar"
    assert (
        repr(simple_grid.tags[1, :1])
        == "BulkTagsSelection(cells=array([Cell(rect=Rectangle, mob=EmptyMobject, "
        "old=EmptyMobject, tags=Tags(foo=bar))],\n      dtype=object))"
    )
