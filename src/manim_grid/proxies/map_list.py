import keyword
from abc import ABC, abstractmethod
from collections.abc import (
    Iterator,
    MutableMapping,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)

from blinker import signal

from manim_grid.helpers import DELETED, MISSING, _Missing
from manim_grid.typing import is_non_string_sequence

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid

IT = TypeVar("IT")  # Internal type
UT = TypeVar("UT")  # User type


class MapBase(ABC, Generic[IT, UT]):
    """The base class in a composite pattern.

    The class hierachy for which this is the base class allows to interact with a single
    dictionary or a list of dictionaries with a similar interface.
    Used in dictionary based proxies such as TagsProxy or ConfigProxy.
    """

    @abstractmethod
    def itermaps(self) -> Iterator["Map[IT, UT]"]:
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        pass

    def _validate_key(self, key: str) -> None:
        if key.startswith("_"):
            raise KeyError(
                f"{type(self).__name__} keys may not start with '_' (got {key!r}.)"
            )
        if keyword.iskeyword(key):
            raise KeyError(f"{type(self).__name__} key {key!r} is a reserved keyword.")
        if not key.isidentifier():
            raise KeyError(
                f"{type(self).__name__} key {key!r} is not a valid Python identifier."
            )

    def _emit_signal(
        self,
        map_: "Map[IT, UT]",
        key: str,
        before: dict[str, IT],
        after: dict[str, IT],
        value: Any,
    ) -> None:
        if map_.signal_name is not None and map_._owner:
            signal(map_.signal_name).send(
                map_._owner,
                owner=map_._owner,
                grid=map_._owner._grid
                if hasattr(map_._owner, "_grid")
                else map_._owner,
                before=before,
                after=after,
                key=key,
                value=value,
            )

    def __getattr__(self, name: str) -> Any:
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_data", "_owner", "_maps"}:
            super().__setattr__(name, value)
            return
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            super().__delattr__(name)
        else:
            del self[name]


class Map(MapBase[IT, UT], MutableMapping[str, UT | _Missing]):
    """The leaf dictionary.

    Note
    ----
    Being based on MutableMapping, popitem pops in FIFO order.

    """

    signal_name: str | None = None

    def __init__(self, owner: "Cell|Grid|None" = None, **data: UT) -> None:
        self._owner = owner
        self._data: dict[str, IT] = {k: self.wrap(v) for k, v in data.items()}

    def itermaps(self) -> Iterator["Map[IT, UT]"]:
        yield self

    def __getitem__(self, key: str) -> UT | _Missing:
        value = self._data.get(key, MISSING)
        if value is MISSING:
            raise KeyError(key)
        return self.unwrap(value)

    def __setitem__(self, key: str, value: UT | _Missing) -> None:
        self._validate_key(key)
        before = dict(self._data)
        internal = self._data.get(key, MISSING)
        self._data[key] = self.wrap(value, internal)
        self._emit_signal(self, key, before, dict(self._data), value)

    def __delitem__(self, key: str) -> None:
        if key not in self._data:
            raise KeyError(key)
        before = dict(self._data)
        del self._data[key]
        self._emit_signal(self, key, before, dict(self._data), DELETED)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def wrap(self, value: UT | _Missing, existing: IT | _Missing = MISSING) -> IT:
        return cast(IT, value)

    def unwrap(self, internal: IT | _Missing) -> UT | _Missing:
        return cast(UT | _Missing, internal)

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self._data.items()]
        return f"{type(self).__name__}({', '.join(parts)})"


class MapList(MapBase[IT, UT], MutableMapping[str, Sequence[UT | _Missing]]):
    """A list of Map with the same interface as a single Map.

    Note
    ----
    Being based on MutableMapping, popitem pops in FIFO order.
    Because __getitem__ does not raise when some child Maps are missing the key,
    returning `MISSING` instead, some methods don't quite work as usual. setdfault is
    overriden for that reason.

    """

    def __init__(self, *maps: Map[IT, UT]) -> None:
        self._maps = list(maps)

    def itermaps(self) -> Iterator["Map[IT, UT]"]:
        yield from self._maps

    def __getitem__(self, key: str) -> list[UT | _Missing]:
        """Return a list of values (one from each Map)."""
        if all(key not in map_._data for map_ in self._maps):
            raise KeyError(key)
        return [map_.unwrap(map_._data.get(key, MISSING)) for map_ in self._maps]

    def __setitem__(self, key: str, value: Sequence[UT | _Missing]) -> None:
        """Set values from a Sequence (one for each Map)."""
        self._validate_key(key)

        if not is_non_string_sequence(value):
            raise TypeError(f"Expected a Sequence. Got {value!r} ({type(value)})")
        if len(value) != len(self._maps):
            raise ValueError(
                f"Expected {len(self._maps)} values for key '{key}', got {len(value)}"
            )

        for map_, val in zip(self._maps, value, strict=True):
            before = dict(map_._data)
            internal = map_._data.get(key, MISSING)
            map_._data[key] = map_.wrap(val, internal)
            self._emit_signal(map_, key, before, dict(map_._data), val)

    def __delitem__(self, key: str) -> None:
        """Delete the key from all Maps."""
        for map_ in self._maps:
            if key in map_._data:
                before = dict(map_._data)
                del map_._data[key]
                self._emit_signal(map_, key, before, dict(map_._data), DELETED)

    def __iter__(self) -> Iterator[str]:
        seen = set()
        for map_ in self._maps:
            for key in map_._data:
                if key not in seen:
                    seen.add(key)
                    yield key

    def __len__(self) -> int:
        return len({key for map_ in self._maps for key in map_})

    def __str__(self) -> str:
        return f"{self._maps}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._maps})"

    def setdefault(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: str, default: Sequence[UT | _Missing]
    ) -> list[UT | _Missing]:
        """Set `key` from `default` Sequence in Maps where it is missing.

        Parameters
        ----------
        key
            The key to set.
        default
            A list of values (one for each Map in the MapList).

        Returns
        -------
        Sequence[UT | _Missing]
            The list of values for this key across all Maps
        """
        if not is_non_string_sequence(default):
            raise TypeError(
                f"Expected a Sequence for default. Got {default!r} ({type(default)})"
            )
        if len(default) != len(self._maps):
            raise ValueError(
                f"Expected {len(self._maps)} values for default, got {len(default)}"
            )

        # Check if key exists in ALL maps
        if all(key in map_._data for map_ in self._maps):
            return self[key]

        # Key missing in at least one map - set default in Maps that miss it
        for map_, val in zip(self._maps, default, strict=True):
            if key not in map_:
                map_[key] = val
        return self[key]
