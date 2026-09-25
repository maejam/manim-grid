from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import manim as m

from .base import ReadableProxy

if TYPE_CHECKING:
    pass


class OldsProxy(ReadableProxy[m.Mobject, m.Group | m.VGroup]):
    """Read-only proxy that exposes the ``old`` attribute of each cell.

    The ``old`` attribute stores the *previous* :class:`manim.Mobject` that was present
    in the cell before the most recent insertion. It is useful for animations that need
    for instance to fade out or transform the former content.

    """

    _attr: str = "old"

    @overload
    def _get_bulk_container_type(self, values: list[m.Mobject]) -> type[m.Group]: ...

    @overload
    def _get_bulk_container_type(self, values: list[m.VMobject]) -> type[m.VGroup]: ...

    def _get_bulk_container_type(
        self, values: Sequence[m.Mobject | m.VMobject]
    ) -> type[m.Group] | type[m.VGroup]:
        if all(isinstance(val, m.VMobject) for val in values):
            return m.VGroup
        return m.Group
