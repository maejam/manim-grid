from typing import (
    TYPE_CHECKING,
)

import manim as m
from manim.typing import Vector3D

from .base import ReadableProxy, WriteableProxy

if TYPE_CHECKING:
    pass


class AlignmentProxy(
    ReadableProxy[Vector3D, list[Vector3D]], WriteableProxy[Vector3D, Vector3D]
):
    """Proxy that provides read-write access to the ``alignment`` attribute.

    NOTE
    ----
    When assigning in bulk (e.g. `grid.alignment[0] = UP`), the same Vector3D object is
    assigned to each targeted cell. Do not mutate one of them unless you know what you
    are doing:

    Example::
        >>> grid = Grid([1]*2, [1]*2)
        >>> grid.alignment[0, :] = UP
        >>> grid.alignment[0, 0][0] = 6
        >>> grid.alignemnt
        [['[6. 1. 0.]' '[6. 1. 0.]']
        ['[0. 0. 0.]' '[0. 0. 0.]']]

    """

    _attr: str = "alignment"
    _bulk_container: type[list[Vector3D] | m.VGroup] = list[Vector3D]
