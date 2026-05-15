import contextlib
import keyword
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import (
    ItemsView,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from typing import TYPE_CHECKING, Any

from blinker import signal

from manim_grid.helpers import DELETED, MISSING, UNSET

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid


class _DictBase(ABC):
    """The base class in a composite pattern.

    The class hierachy for which this is the base class allows to interact with a single
    dictionary or a list of dictionaries with a similar interface.
    Used in dictionary based proxies such as TagsProxy or ConfigProxy.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"data", "owner"}:
            object.__setattr__(self, name, value)
            return
        for item in self.iteritems():
            item[name] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        result = [item.get(name, MISSING) for item in self.iteritems()]
        return result[0] if len(result) == 1 else result

    def __delattr__(self, name: str) -> None:
        for item in self.iteritems():
            with contextlib.suppress(KeyError):
                del item[name]

    @abstractmethod
    def iteritems(self) -> Iterator["_Dict"]: ...


class _Dict(UserDict[str, Any], _DictBase):
    """The leaf dictionary.

    A dict-like class with dot notation attribute access and key validation.

    Attributes
    ----------
    signal_name
        The name of the signal that will be sent from __setitem__ and __delitem__.
        If None, no signal is sent.

    Parameters
    ----------
    dict_
        A dictionnary used to initialize the _Dict keys/values.
    owner
        The Grid or Cell instance this _Dict belongs to. Leave at `None`: it will be
        populated by `Grid.__init__` and `Cell.__init__`.
    **kwargs
        Initial key/value pairs to store. If both `dict_` and `kwargs` set the same
        key, `kwargs` will take precedence.
    """

    signal_name: str | None = None

    def __init__(
        self,
        dict_: Mapping[str, Any] | None = None,
        /,
        *,
        owner: "Cell|Grid|None" = None,
        **kwargs: Any,
    ) -> None:
        self.owner = owner
        super().__init__(dict_, **kwargs)

    def iteritems(self) -> Iterator["_Dict"]:
        yield self

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

    def __setitem__(self, key: str, value: Any) -> None:
        """Validate key and send the signal."""
        from manim_grid.grid import Cell

        self._validate_key(key)
        before = dict(self)
        super().__setitem__(key, value)
        if self.signal_name is not None:
            assert self.owner is not None
            signal(self.signal_name).send(
                self.owner,
                grid=self.owner._grid if isinstance(self.owner, Cell) else self.owner,
                before=before,
                after=dict(self),
                key=key,
                value=value,
            )

    def __delitem__(self, key: str) -> None:
        """Send the signal."""
        from manim_grid.grid import Cell

        before = dict(self)
        super().__delitem__(key)
        if self.signal_name is not None:
            assert self.owner is not None
            signal(self.signal_name).send(
                self.owner,
                grid=self.owner._grid if isinstance(self.owner, Cell) else self.owner,
                before=before,
                after=dict(self),
                key=key,
                value=DELETED,
            )

    def popitem(self) -> tuple[(str, Any)]:
        """Override because UserDict.popitem does not call __delitem__."""
        if not self.data:
            raise KeyError(
                f"popitem(): this {type(self).__name__} dictionary is empty."
            )
        last_key = next(reversed(self.data))
        value = self.data[last_key]

        del self[last_key]
        return last_key, value

    def __repr__(self) -> str:
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.items())
        return f"{type(self).__name__}({attrs})"


class _DictList(list[_Dict], _DictBase):
    """A list of _Dict with the same interface as single _Dict.

    This class partly inherits its behavior from list. As such, the list methods are
    available to manipulate its elements with one exception: `pop`. This is because
    to be able to interact with _DictList as if it were a single _Dict, this class also
    defines the dict methods and forwards them to _Dict. Because `pop` is both a list
    and a dict method, the choice was made to keep the interface consistent with _Dict
    and keep the dict method.
    All dict methods return a list of whatever that method returns for each _Dict entry
    in the _DictList, and mutates the _Dict instances in the same way dict would.

    Parameters
    ----------
    _dicts
        The initial _Dict instances to store in the _DictList. `__init__` is inherited
        from list.
    """

    def iteritems(self) -> Iterator["_Dict"]:
        yield from self

    def update(self, *args: Any, **kwargs: Any) -> None:
        for item in self.iteritems():
            item.update(*args, **kwargs)

    def pop(self, key: str, default: Any = UNSET) -> Any:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        """Pop a key/value pair from all _Dict in the _DictList.

        If no default is passed and keys are missing, it will raise.
        If a default is passed, then this default will be returned in place of the
        missing keys.
        """
        results = []
        # ensure atomicity: all or nothing
        # first, get values
        for item in self.iteritems():
            if default is not UNSET:
                results.append(item.get(key, default))
            else:
                results.append(item[key])

        # everything succeeded -> mutate items
        for item in self.iteritems():
            item.pop(key, default)

        return results

    def popitem(self) -> list[tuple[str, Any]]:
        """Pop the last key/value item from all _Dict in the _DictList.

        Will always raise if any _Dict is empty.
        """
        if any(len(item) == 0 for item in self.iteritems()):
            raise KeyError(
                f"popitem(): at least one empty {type(self).__name__} dictionary in "
                "the selected cells."
            )

        results = []
        for item in self.iteritems():
            results.append(item.popitem())
        return results

    def clear(self) -> None:
        for item in self.iteritems():
            item.clear()

    def setdefault(self, key: str, default: Any = None) -> list[Any]:
        results = []
        for item in self.iteritems():
            results.append(item.setdefault(key, default))
        return results

    def get(self, key: str, default: Any = None) -> list[Any]:
        results = []
        for item in self.iteritems():
            results.append(item.get(key, default))
        return results

    def keys(self) -> list[KeysView[str]]:
        return [item.keys() for item in self.iteritems()]

    def values(self) -> list[ValuesView[Any]]:
        return [item.values() for item in self.iteritems()]

    def items(self) -> list[ItemsView[str, Any]]:
        return [item.items() for item in self.iteritems()]
