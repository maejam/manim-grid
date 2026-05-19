from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Self, overload

from manim.typing import Vector3D

from manim_grid.helpers import MISSING, _Missing

from .base import ReadableProxy
from .map_list import Map, MapList

# NOTE: if overloads explosion becomes unmaintanable, consider using pydantic,
# loose type safety for known keys or disallow dynamic keys and use TypedDict


@dataclass(frozen=True)
class ConfigItem:
    value: Any
    priority: int = 0


class ConfigPriorityMixin:
    def get_priority(self, key: str) -> int | list[int]:
        """Get priority for a key."""
        priorities: list[int] = []
        for map_ in self.itermaps():  # type: ignore[attr-defined]
            if key not in map_._data:
                raise KeyError(f"Key '{key}' not found.")
            priorities.append(map_._data[key].priority)
        if len(priorities) == 1:
            return priorities[0]
        else:
            return priorities

    def set_priority(self, key: str, priority: int) -> Self:
        """Set priority for a key."""
        for map_ in self.itermaps():  # type: ignore[attr-defined]
            if key not in map_._data:
                raise KeyError(f"Key '{key}' not found.")

            old_map = map_._data[key]

            if old_map.priority == priority:
                continue

            new_item = replace(old_map, priority=priority)
            map_._data[key] = new_item

        return self


class Config(ConfigPriorityMixin, Map[ConfigItem, Any]):
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

    @overload
    def __setitem__(self, key: Literal["align"], value: Vector3D) -> None: ...

    @overload
    def __setitem__(self, key: str, value: Any | _Missing) -> None: ...

    def __setitem__(self, key: str, value: Any | _Missing) -> None:
        return super().__setitem__(key, value)

    @property
    def align(self) -> Vector3D:
        return self["align"]

    @align.setter
    def align(self, value: Vector3D) -> None:
        self["align"] = value


class ConfigList(ConfigPriorityMixin, MapList[ConfigItem, Any]):
    @overload
    def __getitem__(self, key: Literal["align"]) -> list[Vector3D]: ...

    @overload
    def __getitem__(self, key: str) -> list[Any | _Missing]: ...

    def __getitem__(self, key: str) -> list[Any | _Missing]:  # type: ignore[misc]
        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: Literal["align"], value: Sequence[Vector3D]) -> None: ...

    @overload
    def __setitem__(self, key: str, value: Sequence[Any | _Missing]) -> None: ...

    def __setitem__(self, key: str, value: Sequence[Any | _Missing]) -> None:
        return super().__setitem__(key, value)

    @property
    def align(self) -> list[Vector3D]:
        return self["align"]

    @align.setter
    def align(self, value: Sequence[Vector3D]) -> None:
        self["align"] = value


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
