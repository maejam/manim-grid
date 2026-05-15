from typing import TYPE_CHECKING

import manim as m

from .base import ReadableProxy

if TYPE_CHECKING:
    pass


class OldsProxy(ReadableProxy[m.Mobject, m.VGroup]):
    """Read-only proxy that exposes the ``old`` attribute of each cell.

    The ``old`` attribute stores the *previous* :class:`manim.Mobject` that was present
    in the cell before the most recent insertion. It is useful for animations that need
    for instance to fade out or transform the former content.

    """

    _attr: str = "old"
    _bulk_container = m.VGroup
