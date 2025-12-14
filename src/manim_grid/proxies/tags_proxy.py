from abc import abstractmethod
from collections import UserDict
from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING, Any, Self, cast, overload

import numpy as np

from manim_grid.typing import BulkIndex, ScalarIndex

from .base import MISSING, ReadableProxy, WriteableProxy

if TYPE_CHECKING:
    from manim_grid.grid import Cell


class Tags(UserDict[str, Any]):
    """Store user-defined tags per cell as attributes."""

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        attrs = ", ".join(f"{key}={value}" for key, value in self.data.items())
        return f"Tags({attrs})"

    def __getattr__(self, name: str) -> Any:
        try:
            return self.data[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"data"}:
            super().__setattr__(name, value)
        else:
            self.data[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self.data[name]
        except KeyError as e:
            raise AttributeError(name) from e


class _TagsSelectionBase:
    """Methods that are identical for scalar and bulk selections.

    Subclasses instances are returned from indexing the ``tags`` argument on a grid
    (i.e. ``grid.tags[...]`` returns :class:`ScalarTagsSelection` or
    :class:`BulkTagSelection` instances).

    See Also
    --------
    manim_grid.proxies.ScalarTagsSelection
    manim_grid.proxies.BulkTagsSelection
    """

    def update(self, **kwargs: Any) -> Self:
        """Add or overwrite attributes on every selected Tags object.

        Parameters
        ----------
        **kwargs
            Attributes to be updated in the form "key=value".

        Returns
        -------
        Self
            The selection itself to allow chaining methods and attributes calls.
        """
        for tags in iter(self):
            tags.data.update(kwargs)
        return self

    def remove(self, *keys: str) -> Self:
        """Delete the given keys from every selected Tags object.

        Parameters
        ----------
        *keys
            Keys to remove from the selected Tags objects.

        Returns
        -------
        Self
            The selection itself to allow chaining methods and attributes calls.
        """
        for tags in iter(self):
            for key in keys:
                tags.data.pop(key, None)
        return self

    def clear(self) -> Self:
        """Remove *all* user-defined attributes from every selected Tags.

        Returns
        -------
        Self
            The selection itself to allow chaining methods and attributes calls.
        """
        for tags in iter(self):
            tags.data.clear()
        return self

    @abstractmethod
    def __iter__(self) -> Generator[Tags]: ...


class ScalarTagsSelection(_TagsSelectionBase):
    """Handle a single Cell selection."""

    __slots__ = ("_cell",)

    def __init__(self, cell: "Cell") -> None:
        self._cell = cell

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cell.tags, name, MISSING)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_cell":
            super().__setattr__(name, value)
            return
        setattr(self._cell.tags, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._cell.tags, name)

    def __iter__(self) -> Generator[Tags]:
        yield self._cell.tags

    def __str__(self) -> str:
        return str(self._cell.tags)

    def __repr__(self) -> str:
        return f"ScalarTagsSelection(cell={self._cell!r})"


class BulkTagsSelection(_TagsSelectionBase):
    """Handle a selection with multiple Cells."""

    __slots__ = ("_cells",)

    def __init__(self, cells: np.ndarray) -> None:
        self._cells = cells

    def __getattr__(self, name: str) -> np.ndarray:
        vec = np.vectorize(
            lambda cell: getattr(cell.tags, name, MISSING), otypes=[object]
        )
        return cast(np.ndarray, vec(self._cells))

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_cells":
            super().__setattr__(name, value)
            return
        for cell in self._cells.flat:
            setattr(cell.tags, name, value)

    def __delattr__(self, name: str) -> None:
        for cell in self._cells.flat:
            delattr(cell.tags, name)

    def __iter__(self) -> Generator[Tags]:
        for cell in self._cells.flat:
            yield cell.tags

    def __str__(self) -> str:
        vec = np.vectorize(lambda cell: str(cell.tags), otypes=[object])
        return str(vec(self._cells))

    def __repr__(self) -> str:
        return f"BulkTagsSelection(cells={self._cells!r})"


class TagsProxy(ReadableProxy[Tags], WriteableProxy[Tags]):
    """Proxy that forwards attribute access to the ``tags`` field of each Cell.

    It returns a _TagsSelection view so that the user can request a given tag or chain
    ``.update/.remove/.clear`` after an indexing operation.

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
    >>> # Update multiple tags on every cell in the first row
    >>> g.tags[0].update(foo="bar", baz=42)
    >>> # Retrieve the ``foo`` flag for the entire grid
    >>> foo_flags = g.tags[:, :].foo
    >>> isinstance(foo_flags, np.ndarray)
    True
    >>> foo_flags.shape
    (2, 3)

    Missing-tag handling
    --------------------
    >>> # Only the first row received the ``baz`` tag
    >>> baz = g.tags[:, :].baz
    >>> baz[0, 0]                     # present: returns the value
    42
    >>> baz[1, 2]                     # absent: returns the sentinel
    '<MISSING>'

    Removing and clearing tags
    ---------------------------
    >>> # Delete the ``foo`` flag from a rectangular block
    >>> g.tags["1":"2", "2":"3"].remove("foo")
    >>> g.tags["1", "2"].foo is MISSING
    True

    >>> # Clear *all* user-defined tags from a masked selection
    >>> g.mobs["1", "1"] = Circle()
    >>> mask = g.mobs.mask(predicate=lambda mob: isinstance(mob, Circle))
    >>> g.tags[mask].clear()
    >>> g.tags["1", "1"].foo is MISSING
    True

    Mixing attribute and method calls
    --------------------------------
    >>> # You can still read/write attributes after a method call
    >>> g.tags["2", "2"].update(priority=5).remove("foo").priority
    5

    See Also
    --------
    manim_grid.proxies.mobs_proxy.MobsProxy,
    manim_grid.proxies.olds_proxy.OldsProxy
    """

    _attr = "tags"

    @overload
    def __getitem__(self, index: ScalarIndex) -> ScalarTagsSelection: ...

    @overload
    def __getitem__(self, index: BulkIndex) -> BulkTagsSelection: ...

    def __getitem__(
        self, index: ScalarIndex | BulkIndex
    ) -> ScalarTagsSelection | BulkTagsSelection:
        return cast(ScalarTagsSelection | BulkTagsSelection, super().__getitem__(index))

    def _postprocess_get(
        self, subarray: "Cell | np.ndarray", **_: Any
    ) -> ScalarTagsSelection | BulkTagsSelection:
        if isinstance(subarray, np.ndarray):
            return BulkTagsSelection(subarray)
        return ScalarTagsSelection(subarray)

    def __setitem__(
        self, index: ScalarIndex | BulkIndex, value: Tags | Mapping[str, Any]
    ) -> None:
        super().__setitem__(index, value)

    def _postprocess_set(
        self,
        subarray: "Cell | np.ndarray",
        value: Tags | Mapping[str, Any],
        **_: Any,
    ) -> None:
        """Replace the ``tags`` attribute on the selected cells.

        Accept a ready-made Tags instance or any mapping that can become one.
        """
        if not isinstance(value, Tags):
            if isinstance(value, Mapping):
                value = Tags(**value)
            else:
                raise TypeError("TagsProxy expects a Tags instance or a mapping.")

        if isinstance(subarray, np.ndarray):
            for cell in subarray.flat:
                setattr(cell, self._attr, value.copy())
        else:
            setattr(subarray, self._attr, value)
