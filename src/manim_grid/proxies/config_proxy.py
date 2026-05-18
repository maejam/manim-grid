from dataclasses import dataclass, replace
from typing import Any, Literal, overload

from manim.typing import Vector3D

from manim_grid.helpers import MISSING, _Missing

from .base import ReadableProxy
from .map_list import Map, MapList


@dataclass
class ConfigItem:
    value: Any
    priority: int = 0


class Config(Map[ConfigItem, Any]):
    def wrap(self, value: Any, existing: ConfigItem | _Missing = MISSING) -> ConfigItem:
        if existing is not MISSING:
            return replace(existing, value=value)
        return ConfigItem(value=value)

    def unwrap(self, internal: ConfigItem | _Missing) -> Any | _Missing:
        if isinstance(internal, _Missing):
            return internal
        return internal.value

    @overload
    def __getitem__(self, key: Literal["align"]) -> Vector3D: ...

    @overload
    def __getitem__(self, key: str) -> Any | _Missing: ...

    def __getitem__(self, key: str) -> Any | _Missing:
        return super().__getitem__(key)


class ConfigList(MapList[ConfigItem, Any]): ...


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
