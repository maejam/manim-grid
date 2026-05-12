from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import manim as m
from blinker import signal
from manim_utils import get_bounds

from .base import ReadableProxy, _DictList

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid


BuiltinModes: TypeAlias = Literal["IDENTITY", "CROP", "SCALE", "STRETCH"]


@signal("cell_updating").connect_via("SCALE")
def scale_mobject(sender: str, key: str, grid: "Grid", cell: "Cell") -> None:
    w, h, _, _ = get_bounds(cell.rect, as_len=True, include_stroke=False)
    w -= cell._grid._margin[0]
    h -= cell._grid._margin[1]
    if cell.mob.width > w:
        cell.mob.scale_to_fit_width(w)
    if cell.mob.height > h:
        cell.mob.scale_to_fit_height(h)


@signal("cell_updating").connect_via("STRETCH")
def stretch_mobject(sender: str, key: str, grid: "Grid", cell: "Cell") -> None:
    w, h, _, _ = get_bounds(cell.rect, as_len=True, include_stroke=False)
    w -= cell._grid._margin[0]
    h -= cell._grid._margin[1]
    if cell.mob.width > w:
        cell.mob.stretch_to_fit_width(w)
    if cell.mob.height > h:
        cell.mob.stretch_to_fit_height(h)


class CellUpdaterContext:
    def __init__(self, cell_updater: "CellUpdater") -> None:
        pass


class CellUpdaterBase(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator["CellUpdater"]: ...

    def __call__(self) -> None:
        for cell_updater in self:
            cell_updater._update(cell_updater)

    def __enter__(self) -> None:
        for cell_updater in self:
            cell_updater.add_updater(cell_updater._update)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        for cell_updater in self:
            cell_updater.remove_updater(cell_updater._update)
        return None


class CellUpdater(CellUpdaterBase, m.Mobject):
    def __init__(self, owner: "Cell") -> None:
        self._owner = owner
        super().__init__(name=f"CellUpdater[{owner.row_index}, {owner.col_index}]")

    def __iter__(self) -> Iterator["CellUpdater"]:
        yield self

    def _update(self, cell_updater: m.Mobject) -> None:
        cell = cast("Cell", cell_updater._owner)
        for key, value in cell.config.items():
            if key == "align":
                continue
            signal("cell_updating").send(value, key=key, grid=cell._grid, cell=cell)
        cell.align_mob(cell.config["align"], cell._grid._margin)


class CellUpdaterList(list[CellUpdater], CellUpdaterBase): ...


class CellsUpdaterProxy(ReadableProxy[CellUpdater, CellUpdaterList]):
    """Proxy that forwards attribute access to the ``config`` field of each Cell.

    It returns a Config or ConfigList object so that the user can request a given config
    option or chain ``.update`` ``setdefault``... after an indexing operation.

    See Also
    --------
    manim_grid.proxies.mobs_proxy.TagsProxy,
    manim_grid.proxies.mobs_proxy.MobsProxy,
    manim_grid.proxies.olds_proxy.OldsProxy

    """

    _attr = "updater"
    _bulk_container: type[list[CellUpdater] | m.VGroup | _DictList] = CellUpdaterList
