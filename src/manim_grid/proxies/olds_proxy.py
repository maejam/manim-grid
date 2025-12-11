from typing import TYPE_CHECKING, Any, cast, overload

import manim as m
import numpy as np

from manim_grid.typing import BulkIndex, ScalarIndex

from .base import ReadableProxy

if TYPE_CHECKING:
    from manim_grid.grid import Cell


class OldsProxy(ReadableProxy[m.Mobject]):
    """Read-only proxy that exposes the ``old`` attribute of each cell.

    The ``old`` attribute stores the *previous* :class:`manim.Mobject` that was present
    in the cell before the most recent insertion. It is useful for animations that need
    for instance to fade out or transform the former content.

    Parameters
    ----------
    grid
        Owning grid instance.
    """

    _attr: str = "old"

    @overload
    def __getitem__(self, index: ScalarIndex) -> m.Mobject: ...

    @overload
    def __getitem__(self, index: BulkIndex) -> list[m.Mobject]: ...

    def __getitem__(
        self, index: ScalarIndex | BulkIndex
    ) -> m.Mobject | list[m.Mobject]:
        return cast(m.Mobject | list[m.Mobject], super().__getitem__(index))

    def _postprocess_get(
        self, subarray: "Cell | np.ndarray", **_: Any
    ) -> m.Mobject | list[m.Mobject]:
        """Return a single Mobject in the scalar case or a list of Mobjects."""
        from manim_grid.grid import Cell

        if isinstance(subarray, Cell):
            return cast(m.Mobject, getattr(subarray, self._attr))

        return [getattr(cell, self._attr) for cell in subarray.flat]
