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
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    overload,
)

import numpy as np
from blinker import signal

from manim_grid.helpers import _UNSET
from manim_grid.typing import BulkIndex, ScalarIndex

from .base import MISSING, ReadableProxy

if TYPE_CHECKING:
    from manim_grid.grid import Cell


class _DeletedSentinel:
    """Sentinel object used to signal that a tag was deleted."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<DELETED>"


DELETED = _DeletedSentinel()


class TagsBase(ABC):
    """The base class in a composite-like pattern.

    Defines the behavior in common between Tags and TagsList. Those 3 classes make it
    possible to interact with Tags or TagsList in the same way.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"data", "cell"}:
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
    cell
        The Cell instance this Tags belongs to. Leave at `None`: it will populated by
        `Grid.__init__` and `TagsProxy.__setitem__`.
    **kwargs
        Initial key/value pairs tags to store. If both `dict_` and `kwargs` set the same
        key, `kwargs` will take precedence.
    """

    def __init__(
        self,
        dict_: Mapping[str, Any] | None = None,
        /,
        *,
        cell: "Cell|None" = None,
        **kwargs: Any,
    ) -> None:
        self.cell = cell
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
        self._validate_key(key)
        before = dict(self)
        super().__setitem__(key, value)
        assert self.cell is not None
        signal("tag_changed").send(
            self.cell,
            grid=self.cell._grid,
            before=before,
            after=dict(self),
            key=key,
            value=value,
        )

    def __delitem__(self, key: str) -> None:
        """Send signal when deleting tags."""
        before = dict(self)
        super().__delitem__(key)
        assert self.cell is not None
        signal("tag_changed").send(
            self.cell,
            grid=self.cell._grid,
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
    to be able to interact with TagsList as if it were a single Tags , this method also
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
            # tags.popitem() is not FIFO
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


class TagsProxy(ReadableProxy[Tags]):
    """Proxy that forwards attribute access to the ``tags`` field of each Cell.

    It returns a Tags or TagsList object so that the user can request a given tag or
    chain ``.update/.remove/.clear``... after an indexing operation.

    Examples
    --------
    >>> from manim_grid import Grid, MISSING
    >>> import numpy as np
    >>> from manim import *
    >>> # Create a simple 2×3 grid
    >>> g = Grid(row_heights=[1, 1], col_widths=[1, 1, 1])

    Basic attribute access
    ----------------------
    >>> # Set a tag on a single cell using attribute syntax
    >>> g.tags[1, 1].foo = "bar"
    >>> g.tags[1, 1].foo
    'bar'

    Bulk assignment and retrieval
    -----------------------------
    >>> # Set the 'foo' tag on all the cells in the first row
    >>> g.tags[0].foo = "bar"
    >>> g.tags[0].foo
    ['bar', 'bar', 'bar']

    >>> # Using the dict methods
    >>> g.tags[0].update(foo="qux", baz=42)
    >>> # Retrieve the ``foo`` flag for the entire grid
    >>> foo_flags = g.tags[:].foo
    >>> isinstance(foo_flags, TagsList)
    True
    >>> len(foo_flags)
    6

    Missing-tag handling
    --------------------
    >>> # Only the first row received the ``baz`` tag
    >>> baz_tag = g.tags[:].baz
    >>> baz[0, 0]                     # present: returns the value
    42
    >>> baz[1, 2]                     # absent: returns the MISSING sentinel
    '<MISSING>'

    NOTE
    ----
    Setting a mutable object as a value will result in a shared object:

    Example::
        >>> grid = Grid([1]*2, [1]*2)
        >>> grid.tags[0, :].mutable = [1, 2]
        >>> grid.tags[0, 0].mutable.append(3)
        >>> grid.tags
        [['Tags(mutable=[1, 2, 3])' 'Tags(mutable=[1, 2, 3])']
        ['Tags(mutable=[1, 2, 3])' 'Tags(mutable=[1, 2, 3])']]


    See Also
    --------
    manim_grid.proxies.mobs_proxy.MobsProxy,
    manim_grid.proxies.olds_proxy.OldsProxy
    """

    _attr = "tags"

    @overload
    def __getitem__(self, index: ScalarIndex) -> Tags: ...

    @overload
    def __getitem__(self, index: BulkIndex) -> TagsList: ...

    def __getitem__(self, index: ScalarIndex | BulkIndex) -> Tags | TagsList:
        return cast(Tags | TagsList, super().__getitem__(index))

    def _postprocess_get(
        self, subarray: "Cell | np.ndarray", **_: Any
    ) -> Tags | TagsList:
        from manim_grid.grid import Cell

        if isinstance(subarray, Cell):
            return cast(Tags, getattr(subarray, self._attr))
        return TagsList(getattr(cell, self._attr) for cell in subarray.flat)
