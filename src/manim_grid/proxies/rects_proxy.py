from typing import TYPE_CHECKING, Any, cast, overload

import manim as m
import numpy as np

from manim_grid.typing import BulkIndex, ScalarIndex

from .base import ReadableProxy

if TYPE_CHECKING:
    from manim_grid.grid import Cell


class RectsProxy(ReadableProxy[m.Mobject]):
    """Read-only proxy that exposes the ``rect`` attribute of each cell.

    The ``rect`` attribute stores the Rectangles defining the boundaries of each cell.
    Those Rectangles are also accessible via ``Grid.frame`` which returns a ``VGroup``.
    This proxy returns a numpy array which gives more flexibility to target specific
    Rectangles.

    Examples
    --------
    >>> # Color every other row.
    >>> import manim as m
    >>> from manim_grid import Grid
    >>> g = Grid([1]*3, [1]*5)
    >>> g.rects[::2].set_color(m.WHITE).set_opacity(0.3)
    >>> self.add(g)

    Parameters
    ----------
    grid
        Owning grid instance.
    """

    _attr: str = "rect"

    @overload
    def __getitem__(self, index: ScalarIndex) -> m.Mobject: ...

    @overload
    def __getitem__(self, index: BulkIndex) -> m.VGroup: ...

    def __getitem__(self, index: ScalarIndex | BulkIndex) -> m.Mobject | m.VGroup:
        return cast(m.Mobject | m.VGroup, super().__getitem__(index))

    def _postprocess_get(
        self, subarray: "Cell | np.ndarray", **_: Any
    ) -> m.Rectangle | m.VGroup:
        """Return a single Rectangle in the scalar case or a VGroup of Rectangles."""
        from manim_grid.grid import Cell

        if isinstance(subarray, Cell):
            return cast(m.Rectangle, getattr(subarray, self._attr))

        return m.VGroup(getattr(cell, self._attr) for cell in subarray.flat)
