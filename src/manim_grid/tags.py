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

from manim_grid.helpers import _UNSET
from manim_grid.proxies.base import MISSING

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid


class _DeletedSentinel:
    """Sentinel object used to signal that a tag was deleted."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<DELETED>"


DELETED = _DeletedSentinel()


class TagsBase(ABC):
    """The base class in a composite-like pattern.

    Defines the behavior in common between Tags and TagsList. Those 3 classes make it
    possible to interact with Tags or TagsList with a similar interface.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"data", "owner"}:
            object.__setattr__(self, name, value)
            return
        for tags in self.iter_tags():
            tags[name] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        result = [tags.get(name, MISSING) for tags in self.iter_tags()]
        return result[0] if len(result) == 1 else result

    def __delattr__(self, name: str) -> None:
        for tags in self.iter_tags():
            with contextlib.suppress(KeyError):
                del tags[name]

    @abstractmethod
    def iter_tags(self) -> Iterator["Tags"]: ...


class Tags(UserDict[str, Any], TagsBase):
    """Store user-defined tags.

    A dict-like class with dot notation attribute access and key validation.

    Parameters
    ----------
    dict_
        A dictionnary used to initialize the Tags keys/values.
    owner
        The Grid or Cell instance this Tags belongs to. Leave at `None`: it will be
        populated by `Grid.__init__` and `Cell.__init__`.
    **kwargs
        Initial key/value pairs tags to store. If both `dict_` and `kwargs` set the same
        key, `kwargs` will take precedence.
    """

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

    def iter_tags(self) -> Iterator["Tags"]:
        yield self

    def _validate_key(self, key: str) -> None:
        if key.startswith("_"):
            raise KeyError(f"Tag keys may not start with '_' (got {key!r}.)")

        if keyword.iskeyword(key):
            raise KeyError(f"Tag key {key!r} is a reserved keyword.")

        if not key.isidentifier():
            raise KeyError(f"Tag key {key!r} is not a valid Python identifier.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Send signal when setting tags."""
        from manim_grid.grid import Cell

        self._validate_key(key)
        before = dict(self)
        super().__setitem__(key, value)
        assert self.owner is not None
        signal("tag_changed").send(
            self.owner,
            grid=self.owner._grid if isinstance(self.owner, Cell) else self.owner,
            before=before,
            after=dict(self),
            key=key,
            value=value,
        )

    def __delitem__(self, key: str) -> None:
        """Send signal when deleting tags."""
        from manim_grid.grid import Cell

        before = dict(self)
        super().__delitem__(key)
        assert self.owner is not None
        signal("tag_changed").send(
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
            raise KeyError("popitem(): this 'Tags' dictionary is empty.")
        last_key = next(reversed(self.data))
        value = self.data[last_key]

        del self[last_key]
        return last_key, value

    def __repr__(self) -> str:
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.items())
        return f"Tags({attrs})"


class TagsList(list[Tags], TagsBase):
    """A list of Tags with the same interface as single Tags.

    This class partly inherits its behavior from list. As such, the list methods are
    available to manipulate its elements with one exception: `pop`. This is because
    to be able to interact with TagsList as if it were a single Tags , this class also
    defines the dict methods and forwards them to Tags. Because `pop` is both a list
    and a dict method, the choice was made to keep the interface consistent with Tags
    and keep the dict method.
    All dict methods return a list of whatever that method returns for each Tags entry
    in the TagsList, and mutates the Tags instances in the same way dict would.

    Parameters
    ----------
    tags
        The initial Tags instances to store in the TagsList. `__init__` is inherited
        from list.
    """

    def iter_tags(self) -> Iterator["Tags"]:
        yield from self

    def update(self, *args: Any, **kwargs: Any) -> None:
        for tags in self.iter_tags():
            tags.update(*args, **kwargs)

    def pop(self, key: str, default: Any = _UNSET) -> Any:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        """Pop a key/value pair from all Tags in the TagsList.

        If no default is passed and keys are missing, it will raise.
        If a default is passed, then this default will be returned in place of the
        missing keys.
        """
        results = []
        # ensure atomicity: all or nothing
        # first, get values
        for tags in self.iter_tags():
            if default is not _UNSET:
                results.append(tags.get(key, default))
            else:
                results.append(tags[key])

        # everything succeeded -> mutate tags
        for tags in self.iter_tags():
            tags.pop(key, default)

        return results

    def popitem(self) -> list[tuple[str, Any]]:
        """Pop the last key/value item from all Tags in the TagsList.

        Will always raise on empty Tags dictionaries.
        """
        if any(len(tags) == 0 for tags in self.iter_tags()):
            raise KeyError(
                "popitem(): at least one empty `Tags` dictionary in the selected cells."
            )

        results = []
        for tags in self.iter_tags():
            results.append(tags.popitem())
        return results

    def clear(self) -> None:
        for tags in self.iter_tags():
            tags.clear()

    def setdefault(self, key: str, default: Any = None) -> list[Any]:
        results = []
        for tags in self.iter_tags():
            results.append(tags.setdefault(key, default))
        return results

    def get(self, key: str, default: Any = None) -> list[Any]:
        results = []
        for tags in self.iter_tags():
            results.append(tags.get(key, default))
        return results

    def keys(self) -> list[KeysView[str]]:
        return [tags.keys() for tags in self.iter_tags()]

    def values(self) -> list[ValuesView[Any]]:
        return [tags.values() for tags in self.iter_tags()]

    def items(self) -> list[ItemsView[str, Any]]:
        return [tags.items() for tags in self.iter_tags()]
