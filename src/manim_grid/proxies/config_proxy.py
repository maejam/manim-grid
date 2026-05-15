from .base import ReadableProxy
from .dict_list import _Dict, _DictList


class Config(_Dict): ...


class ConfigList(_DictList): ...


class ConfigProxy(ReadableProxy[Config, ConfigList]):
    """Proxy that forwards attribute access to the ``config`` field of each Cell.

    It returns a Config or ConfigList object so that the user can request a given config
    option or chain ``.update`` ``setdefault``... after an indexing operation.

    See Also
    --------
    manim_grid.proxies.mobs_proxy.TagsProxy,
    manim_grid.proxies.mobs_proxy.MobsProxy,
    manim_grid.proxies.olds_proxy.OldsProxy

    """

    _attr = "config"
    _bulk_container = ConfigList
