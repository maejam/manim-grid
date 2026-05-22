import manim as m
import pytest

from manim_grid import Grid
from manim_grid.proxies.config_proxy import Config, ConfigList


def test_configproxy_getitem(simple_grid: Grid):
    assert simple_grid.config[0, 0] == {"align": m.ORIGIN, "mode": "NONE"}
    assert isinstance(simple_grid.config[0, 0], Config)
    simple_grid.config[:].update({"align": [m.UP] * 6})
    assert simple_grid.config[0, :] == {"align": [m.UP] * 3, "mode": ["NONE"] * 3}
    assert isinstance(simple_grid.config[0, :], ConfigList)
    assert simple_grid.config[0, 0] is not simple_grid.config[0, 1]


def test_get_set_priority(simple_grid: Grid):
    assert simple_grid.config[0, 0].get_priority("align") == 100
    assert simple_grid.config[0, 0].set_priority("align", 50)
    assert simple_grid.config[0, 0].get_priority("align") == 50

    assert simple_grid.config[0].get_priority("mode") == [0] * 3
    assert simple_grid.config[0].set_priority("mode", 50)
    assert simple_grid.config[0].get_priority("mode") == [50] * 3


def test_get_set_priority_unknown_key_raises(simple_grid: Grid):
    with pytest.raises(KeyError, match="dne"):
        simple_grid.config[0, 0].get_priority("dne")
    with pytest.raises(KeyError, match="dne"):
        assert simple_grid.config[0, 0].set_priority("dne", 50)

    with pytest.raises(KeyError, match="dne"):
        simple_grid.config[0].get_priority("dne")
    with pytest.raises(KeyError, match="dne"):
        assert simple_grid.config[0].set_priority("dne", 50)


def test_sort_by_priority(simple_grid: Grid):
    simple_grid.config[0, 0].a = 1
    simple_grid.config[0, 0].b = 2
    simple_grid.config[0, 0].set_priority("a", 10)
    simple_grid.config[0, 0].set_priority("b", 5)
    prio = simple_grid.config[0, 0].sort_by_priority()
    assert isinstance(prio, dict)
    assert prio == {"mode": "NONE", "b": 2, "a": 1, "align": m.ORIGIN}
    assert list(prio.keys()) == ["mode", "b", "a", "align"]
