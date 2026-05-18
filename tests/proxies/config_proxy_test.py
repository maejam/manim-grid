import manim as m

from manim_grid import Grid
from manim_grid.proxies.config_proxy import Config, ConfigList


def test_configproxy_getitem(simple_grid: Grid):
    assert simple_grid.config[0, 0] == {"align": m.ORIGIN, "mode": "NONE"}
    assert isinstance(simple_grid.config[0, 0], Config)
    simple_grid.config[:].update({"align": [m.UP] * 6})
    assert simple_grid.config[0, :] == {"align": [m.UP] * 3, "mode": ["NONE"] * 3}
    assert isinstance(simple_grid.config[0, :], ConfigList)
    assert simple_grid.config[0, 0] is not simple_grid.config[0, 1]
