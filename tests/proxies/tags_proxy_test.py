from manim_grid import Grid
from manim_grid.proxies.tags_proxy import Tags, TagsList


def test_tagsproxy_getitem(simple_grid: Grid):
    assert simple_grid.tags[0, 0] == {}
    assert isinstance(simple_grid.tags[0, 0], Tags)
    simple_grid.tags[:].update({"baz": [42] * 6})
    assert simple_grid.tags[0, :] == {"baz": [42] * 3}
    assert isinstance(simple_grid.tags[0, :], TagsList)
    assert simple_grid.tags[0, 0] is not simple_grid.tags[0, 1]
