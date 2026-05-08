from typing import TYPE_CHECKING

import manim as m

from .base import ReadableProxy, _DictList

if TYPE_CHECKING:
    pass


class RectsProxy(ReadableProxy[m.Rectangle, m.VGroup]):
    """Read-only proxy that exposes the ``rect`` attribute of each cell.

    The ``rect`` attribute stores the Rectangles defining the boundaries of each cell.
    Those Rectangles are also accessible via ``Grid.lattice`` which returns a
    ``VGroup`` containing all the rectangles.
    This proxy returns a VGroup containing only the targeted Cells Rectangles.

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
        The Grid instance.
    """

    _attr: str = "rect"
    _bulk_container: type[list[m.Rectangle] | m.VGroup | _DictList] = m.VGroup
