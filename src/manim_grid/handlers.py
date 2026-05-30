from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import manim as m
from blinker import ANY, signal
from manim_utils import clip_vmobject, get_bounds

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid


def warn_on_group(sender: Any, mob: m.Mobject, **_: Any) -> None:
    """Log a warning when a (V)Group is added or removed."""
    if isinstance(mob, (m.Group, m.VGroup)):
        m.logger.warning("A Group was added/removed: %s", mob)


def align_mobject(sender: Any, cell: "Cell", **kwargs: Any) -> None:
    """Align a mobject inside its Cell Rectangle."""
    vec = kwargs.get("value", cell.config.align)
    cell.mob.move_to(cell.rect, aligned_edge=vec).shift(-vec * cell._grid._margin)


def _max_width_height(cell: "Cell") -> tuple[float, float]:
    """Return a tuple (max_width, max_height).

    The mobject in a Cell should not exceed those dimensions to be considered inside
    its Rectangle.
    """
    w, h, _, _ = get_bounds(cell.rect, as_len=True, include_stroke=False)
    w -= cell._grid._margin[0]
    h -= cell._grid._margin[1]
    return w, h


def _make_mobcopy(cell: "Cell") -> m.Mobject:
    """Copy the original mobject before destructive operations.

    The initial mobject is backed-up in the `_mobcopy` attribute. A `reset_mob` method
    is also attached to the mobject.
    """
    if not hasattr(cell.mob, "_mobcopy"):
        cell.mob.__dict__["_mobcopy"] = cell.mob.copy()

    def reset_mob() -> None:
        cell.mob.become(cell.mob.__dict__["_mobcopy"])

    cell.mob.__dict__["reset_mob"] = reset_mob
    return cast(m.Mobject, cell.mob.__dict__["_mobcopy"])


def scale_mobject(
    sender: str, key: str, value: str, grid: "Grid", cell: "Cell"
) -> None:
    """Scale a mobject that is bigger than its Cell Rectangle to fit inside it."""
    if value != "SCALE":
        return
    w, h = _max_width_height(cell)
    mobcopy = _make_mobcopy(cell)
    result = mobcopy.copy()
    if mobcopy.width > w:
        result.scale_to_fit_width(w)
    if mobcopy.height > h:
        result.scale_to_fit_height(h)
    cell.mob.become(result)


def stretch_mobject(
    sender: str, key: str, value: str, grid: "Grid", cell: "Cell"
) -> None:
    """Stretch a mobject that is bigger than its Cell Rectangle to fit inside it."""
    if value != "STRETCH":
        return
    w, h = _max_width_height(cell)
    mobcopy = _make_mobcopy(cell)
    result = mobcopy.copy()
    if mobcopy.width > w:
        result.stretch_to_fit_width(w)
    if mobcopy.height > h:
        result.stretch_to_fit_height(h)
    cell.mob.become(result)


def crop_mobject(sender: str, key: str, value: str, grid: "Grid", cell: "Cell") -> None:
    """Crop a mobject that is bigger than its Cell Rectangle to fit inside it."""
    if value != "CROP":
        return
    w, h = _max_width_height(cell)
    mobcopy = _make_mobcopy(cell)
    if not isinstance(mobcopy, m.VMobject):
        raise TypeError("Can only be used with VMobjects.")
    if mobcopy.width > w or mobcopy.height > h:
        result = clip_vmobject(mobcopy, cell.rect)
        cell.mob.become(result)
    else:
        cell.mob.become(mobcopy)


ConnectionTuple: TypeAlias = tuple[str, Callable[..., Any], Any]

default_connections: dict[str, ConnectionTuple] = {
    # General
    "warn_on_group_added": ("mob_added", warn_on_group, ANY),
    "warn_on_group_removed": ("mob_removed", warn_on_group, ANY),
    # Cell align
    "align_on_mob_inserted": ("mob_inserted", align_mobject, ANY),
    "align_on_cell_updating": ("cell_updating", align_mobject, "align"),
    # Cell mode
    "scale_on_cell_updating": ("cell_updating", scale_mobject, "mode"),
    "stretch_on_cell_updating": ("cell_updating", stretch_mobject, "mode"),
    "crop_on_cell_updating": ("cell_updating", crop_mobject, "mode"),
}


class HandlerManager:
    """Define default handlers that are connected when instantiating a Grid.

    This class in instantiated during :meth:`Grid.__init__`. The instance is accessible
    through `Grid.handlers`. Individual handlers can be connected/disconnected by name
    using the dedicated methods.

    Parameters
    ----------
    connections
        A Mapping from a string name identifier to a tuple
        (signal_name, handler_callable, sender) that represent the connections that
        will be established when instantiating the HandlerManager.
    """

    def __init__(self, connections: Mapping[str, ConnectionTuple]) -> None:
        self.connections = connections
        for connection in self.connections.values():
            signal(connection[0]).connect(connection[1], sender=connection[2])
        self._enabled = dict.fromkeys(connections, True)

    def enable(self, *names: str) -> None:
        """Enable connections by name."""
        for name in names:
            if name in self.connections:
                connection = self.connections[name]
                self._enabled[name] = True
                signal(connection[0]).connect(connection[1], sender=connection[2])

    def disable(self, *names: str) -> None:
        """Disable connections by name."""
        for name in names:
            if name in self.connections:
                connection = self.connections[name]
                self._enabled[name] = False
                signal(connection[0]).disconnect(connection[1], sender=connection[2])

    def enable_all(self) -> None:
        """Enable all connections."""
        for connection in self.connections.values():
            signal(connection[0]).connect(connection[1], sender=connection[2])
        self._enabled = dict.fromkeys(self._enabled, True)

    def disable_all(self) -> None:
        """Disable all connections."""
        for connection in self.connections.values():
            signal(connection[0]).disconnect(connection[1], sender=connection[2])
        self._enabled = dict.fromkeys(self._enabled, False)

    def __str__(self) -> str:
        enabled = [
            key + f" <{self.connections[key][2]}>"
            for key in self._enabled
            if self._enabled[key]
        ]
        disabled = [
            key + f" <{self.connections[key][2]}>"
            for key in self._enabled
            if not self._enabled[key]
        ]
        return f"Handlers[enabled: {enabled} | disabled: {disabled}]"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.connections})"
