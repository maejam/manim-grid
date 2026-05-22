import manim as m
import pytest

from manim_grid.grid import Cell, Grid
from manim_grid.proxies.base import MISSING
from manim_grid.proxies.tags_proxy import Tags, TagsList


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
@pytest.fixture
def cell(simple_grid: Grid):
    return Cell(simple_grid, m.Rectangle(), 1, 2)


# ----------------------------------------------------------------------
# Map / MapList through TagsProxy
# ----------------------------------------------------------------------
def test_tags_str(cell: Cell):
    tags = Tags(foo=1, bar=2, owner=cell)
    assert str(tags) == "Tags(foo=1, bar=2)"


def test_tagslist_str(cell: Cell):
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
    with pytest.raises(KeyError, match="foo"):
        assert simple_grid.tags[1, 1].foo is MISSING
    with pytest.raises(KeyError, match="baz"):
        assert simple_grid.tags[1, 1].baz is MISSING


def test_tagslist_setattr_getattr_and_delattr(simple_grid: Grid):
    assert simple_grid.tags[1, :] == {}
    simple_grid.tags[1, :-1].foo = ["bar"] * 2
    assert simple_grid.tags[1, :] == {"foo": ["bar", "bar", MISSING]}
    del simple_grid.tags[1, 1:].foo
    assert simple_grid.tags[1, :] == {"foo": ["bar", MISSING, MISSING]}
    simple_grid.tags[1, 0].baz = 42
    assert simple_grid.tags[1, :] == {
        "foo": ["bar", MISSING, MISSING],
        "baz": [42, MISSING, MISSING],
    }
    del simple_grid.tags[1, 0].foo
    assert simple_grid.tags[1, :] == {"baz": [42, MISSING, MISSING]}


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
    simple_grid.tags[0].update(foo=["bar"] * 3)
    simple_grid.tags[0].update(foo=["qux"] * 3, baz=[42] * 3)
    assert simple_grid.tags[0, 0] == simple_grid.tags[0, 1] == {"foo": "qux", "baz": 42}


def test_tagslist_pop(simple_grid: Grid):
    simple_grid.tags[:].update({"foo": ["bar"] * 6, "baz": [42] * 6})
    foo = simple_grid.tags[0].pop("foo")
    assert foo == ["bar"] * 3
    with pytest.raises(KeyError, match="foo"):
        _ = simple_grid.tags[0].foo
    assert simple_grid.tags[0].baz == [42] * 3


def test_tagslist_popitem(simple_grid: Grid):
    # popitem removes first element in MutableMapping
    simple_grid.tags[:].update({"foo": ["bar"] * 6, "baz": [42] * 6})
    res = simple_grid.tags[0].popitem()
    assert res == ("foo", ["bar"] * 3)
    assert simple_grid.tags[0] == {"baz": [42] * 3}
    assert simple_grid.tags[1] == {"foo": ["bar"] * 3, "baz": [42] * 3}


def test_tagslist_clear(simple_grid: Grid):
    simple_grid.tags[:].update({"foo": ["bar"] * 6, "baz": [42] * 6})
    simple_grid.tags[0, :2].clear()
    assert simple_grid.tags[0, :2] == {}
    assert simple_grid.tags[0, -1] == {"foo": "bar", "baz": 42}


def test_tags_setdefault(simple_grid: Grid):
    simple_grid.tags[0, 0].update({"foo": "bar", "baz": 42})
    default = simple_grid.tags[0, 0].setdefault("foo", "qux")
    assert default == "bar"
    default = simple_grid.tags[0, 1].setdefault("foo", "qux")
    assert default == "qux"


def test_tagslist_setdefault(simple_grid: Grid):
    simple_grid.tags[0].update({"foo": ["bar"] * 3, "baz": [42] * 3})
    default = simple_grid.tags[0].setdefault("foo", ["qux"] * 3)
    assert default == ["bar", "bar", "bar"]
    assert simple_grid.tags[:] == {
        "foo": ["bar", "bar", "bar", MISSING, MISSING, MISSING],
        "baz": [42, 42, 42, MISSING, MISSING, MISSING],
    }

    default = simple_grid.tags[:].setdefault("foo", ["qux"] * 6)
    assert default == ["bar", "bar", "bar", "qux", "qux", "qux"]
    assert simple_grid.tags[0, 0].foo == "bar"
    assert simple_grid.tags[1, 0].foo == "qux"


def test_tagslist_get(simple_grid: Grid):
    simple_grid.tags[0].update({"foo": ["bar"] * 3, "baz": [42] * 3})
    assert simple_grid.tags[0].get("foo") == ["bar"] * 3
    assert simple_grid.tags[0, 0] == {"foo": "bar", "baz": 42}
    assert simple_grid.tags[1].get("foo") is None
    assert simple_grid.tags[1].get("foo", ["foofoo"] * 3) == ["foofoo"] * 3
    assert simple_grid.tags[1, 0] == {}


def test_tagslist_keys(simple_grid: Grid):
    simple_grid.tags[0, :-1].update({"foo": ["bar"] * 2, "baz": [42] * 2})
    assert list(simple_grid.tags[0].keys()) == ["foo", "baz"]


def test_tagslist_values(simple_grid: Grid):
    simple_grid.tags[0, :-1].update({"foo": ["bar"] * 2, "baz": [42] * 2})
    assert list(simple_grid.tags[0].values()) == [
        ["bar", "bar", MISSING],
        [42, 42, MISSING],
    ]


def test_tagslist_items(simple_grid: Grid):
    simple_grid.tags[0, :-1].update({"foo": ["bar"] * 2, "baz": [42] * 2})
    assert list(simple_grid.tags[0].items()) == [
        ("foo", ["bar", "bar", MISSING]),
        ("baz", [42, 42, MISSING]),
    ]
