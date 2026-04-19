import manim as m
import pytest

from manim_grid.grid import Cell, Grid
from manim_grid.tags import MISSING, Tags, TagsList


@pytest.fixture
def cell(simple_grid: Grid):
    return Cell(simple_grid, m.Rectangle(), 1, 2)


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
