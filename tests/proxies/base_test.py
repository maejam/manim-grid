import manim as m
import numpy as np
import pytest

from manim_grid.grid import Cell, EmptyMobject, Grid
from manim_grid.proxies.base import MISSING
from manim_grid.proxies.tags_proxy import Tags, TagsList


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


@pytest.fixture
def cell(simple_grid: Grid):
    return Cell(simple_grid, m.Rectangle(), 1, 2)


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


# ----------------------------------------------------------------------
# _Dict / _DictList through TagsProxy
# ----------------------------------------------------------------------
def test_tags_str(cell: Cell):
    tags = Tags(foo=1, bar=2, owner=cell)
    assert str(tags) == "Tags(foo=1, bar=2)"


def test_tags_list_str(cell: Cell):
    tags = TagsList([Tags(one=1, two=2, owner=cell), Tags(three=3, owner=cell)])
    assert str(tags) == "[Tags(one=1, two=2), Tags(three=3)]"


def test_tags_repr(cell: Cell):
    tags = Tags(one=1, two=2, owner=cell)
    assert repr(tags) == "Tags(one=1, two=2)"


def test_tagslist_repr(cell: Cell):
    tags = TagsList([Tags(one=1, two=2, owner=cell), Tags(three=3, owner=cell)])
    assert str(tags) == "[Tags(one=1, two=2), Tags(three=3)]"


def test_tags_setattr_getattr_and_delattr(simple_grid: Grid):
    assert simple_grid.tags[1, 1] == {}
    simple_grid.tags[1, 1].foo = "bar"
    assert simple_grid.tags[1, 1].foo == "bar"
    del simple_grid.tags[1, 1].foo
    assert simple_grid.tags[1, 1].foo is MISSING
    assert simple_grid.tags[1, 1].baz is MISSING
    del simple_grid.tags[1, 1].foo
    assert simple_grid.tags[1, 1].foo is MISSING


def test_tagslist_setattr_getattr_and_delattr(simple_grid: Grid):
    assert simple_grid.tags[1, :] == [{}, {}, {}]
    simple_grid.tags[1, :-1].foo = "bar"
    assert simple_grid.tags[1, :] == [{"foo": "bar"}, {"foo": "bar"}, {}]
    del simple_grid.tags[1, 1:].foo
    assert simple_grid.tags[1, :] == [{"foo": "bar"}, {}, {}]
    simple_grid.tags[1, 0].baz = 42
    assert simple_grid.tags[1, :] == [{"foo": "bar", "baz": 42}, {}, {}]
    del simple_grid.tags[1, 0].foo
    assert simple_grid.tags[1, :] == [{"baz": 42}, {}, {}]


def test_tags_validate_key(simple_grid: Grid, cell: Cell):
    with pytest.raises(KeyError, match="may not start with '_'"):
        simple_grid.tags[0, 0]._foo = "bar"

    with pytest.raises(KeyError, match="is a reserved keyword"):
        setattr(simple_grid.tags[0, 0], "class", 1)

    with pytest.raises(KeyError, match="is not a valid Python identifier"):
        setattr(simple_grid.tags[0, 0], "1foo", 2)


def test_tagslist_validate_key(simple_grid: Grid, cell: Cell):
    with pytest.raises(KeyError, match="may not start with '_'"):
        simple_grid.tags[0, :]._foo = "bar"

    with pytest.raises(KeyError, match="is a reserved keyword"):
        setattr(simple_grid.tags[0, :], "def", 1)

    with pytest.raises(KeyError, match="is not a valid Python identifier"):
        setattr(simple_grid.tags[0, :], "1foo", 1)


def test_tagslist_update(simple_grid: Grid):
    simple_grid.tags[0].update(foo="bar")
    simple_grid.tags[0].update(foo="qux", baz=42)
    assert simple_grid.tags[0, 0] == simple_grid.tags[0, 1] == {"foo": "qux", "baz": 42}


def test_tagslist_pop(simple_grid: Grid):
    simple_grid.tags[:].update({"foo": "bar", "baz": 42})
    foo = simple_grid.tags[0].pop("foo")
    assert foo == ["bar"] * 3
    assert simple_grid.tags[0].foo == [MISSING] * 3
    assert simple_grid.tags[0].baz == [42] * 3


def test_tagslist_popitem(simple_grid: Grid):
    simple_grid.tags[:].update({"foo": "bar", "baz": 42})
    res = simple_grid.tags[0].popitem()
    assert res == [("baz", 42)] * 3
    assert simple_grid.tags[0] == [{"foo": "bar"}] * 3
    assert simple_grid.tags[1] == [{"foo": "bar", "baz": 42}] * 3


def test_tagslist_clear(simple_grid: Grid):
    simple_grid.tags[:].update({"foo": "bar", "baz": 42})
    simple_grid.tags[0, :2].clear()
    assert simple_grid.tags[0, :2] == [{}] * 2
    assert simple_grid.tags[0, -1] == {"foo": "bar", "baz": 42}


def test_tagslist_setdefault(simple_grid: Grid):
    simple_grid.tags[0].update({"foo": "bar", "baz": 42})
    simple_grid.tags[:].setdefault("foo", "qux")
    assert simple_grid.tags[0, 0].foo == "bar"
    assert simple_grid.tags[1, 0].foo == "qux"


def test_tagslist_get(simple_grid: Grid):
    simple_grid.tags[0].update({"foo": "bar", "baz": 42})
    assert simple_grid.tags[0].get("foo") == ["bar"] * 3
    assert simple_grid.tags[0, 0] == {"foo": "bar", "baz": 42}
    assert simple_grid.tags[1].get("foo") == [None] * 3
    assert simple_grid.tags[1].get("foo", "foofoo") == ["foofoo"] * 3
    assert simple_grid.tags[1, 0] == {}


def test_tagslist_keys(simple_grid: Grid):
    simple_grid.tags[0].update({"foo": "bar", "baz": 42})
    assert list(simple_grid.tags[0].keys()[0]) == ["foo", "baz"]
    assert list(simple_grid.tags[:].keys()[-1]) == []


def test_tagslist_values(simple_grid: Grid):
    simple_grid.tags[0].update({"foo": "bar", "baz": 42})
    assert list(simple_grid.tags[:].values()[0]) == ["bar", 42]
    assert list(simple_grid.tags[:].values()[-1]) == []


def test_tagslist_items(simple_grid: Grid):
    simple_grid.tags[0].update({"foo": "bar", "baz": 42})
    assert list(simple_grid.tags[0].items()[0]) == [("foo", "bar"), ("baz", 42)]
    assert list(simple_grid.tags[:].items()[-1]) == []
