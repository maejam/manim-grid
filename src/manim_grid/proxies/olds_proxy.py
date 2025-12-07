from typing import Never

import manim as m

from .base import ReadableProxy


class OldsProxy(ReadableProxy[Never, Never, m.Mobject]):
    """Read-only proxy that exposes the ``old`` attribute of each cell.

    The ``old`` attribute stores the *previous* :class:`manim.Mobject` that was present
    in the cell before the most recent insertion. It is useful for animations that need
    for instance to fade out or transform the former content.

    Parameters
    ----------
    grid
        Owning grid instance.
    attr
        Must be ``"old"`` - the name of the attribute this proxy reads.
    """

    ...
