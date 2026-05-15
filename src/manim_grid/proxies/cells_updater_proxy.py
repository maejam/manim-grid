from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import manim as m
from blinker import signal
from manim_utils import get_bounds

from .base import ReadableProxy
from .config_proxy import Config

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


class CellUpdaterBase(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator["CellUpdater"]: ...

    def __call__(self, *args: str, **kwargs: str) -> None:
        for cell_updater in self:
            options = cell_updater._merge_config(*args, **kwargs)
            cell_updater._update(cell_updater, **options)

    def run(self, **kwargs: Any) -> "_CMDecoWrapper":
        return _CMDecoWrapper(self, **kwargs)

    def _attach_updaters(self, **options: Any) -> None:
        for cell_updater in self:
            options_ = cell_updater._merge_config(**options)
            cell_updater._updater = partial(cell_updater._update, **options_)
            cell_updater.add_updater(cell_updater._updater)

    def _detach_updaters(self) -> None:
        for cell_updater in self:
            if cell_updater._updater is not None:
                cell_updater.remove_updater(cell_updater._updater)


class CellUpdater(CellUpdaterBase, m.Mobject):
    def __init__(self, owner: "Cell") -> None:
        self._owner = owner
        self._updater: Callable[[m.Mobject], None] | None = None
        super().__init__(name=f"CellUpdater[{owner.row_index}, {owner.col_index}]")

    def __iter__(self) -> Iterator["CellUpdater"]:
        yield self

    def _merge_config(self, *args: str, **kwargs: Any) -> Config | dict[str, Any]:
        config = self._owner.config
        if not args and not kwargs:
            return config
        d = (
            {key: value for key, value in config.items() if key in args}
            if args
            else config
        )
        d.update(kwargs)
        return d

    def _update(self, cell_updater: m.Mobject, **options: Any) -> None:
        cell = cast("Cell", cell_updater._owner)
        for key, value in options.items():
            if key == "align":
                continue
            # print("sending", value, key, cell._grid, cell)
            signal("cell_updating").send(value, key=key, grid=cell._grid, cell=cell)
        cell.align_mob(cell.config["align"], cell._grid._margin)


class CellUpdaterList(list[CellUpdater], CellUpdaterBase): ...


class _CMDecoWrapper:
    def __init__(self, parent: CellUpdaterBase, **options: Any) -> None:
        self._parent = parent
        self._options = options

    def __enter__(self) -> None:
        self._parent._attach_updaters(**self._options)

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> Literal[False]:
        self._parent._detach_updaters()
        return False

    def __call__(self, func: Callable[..., Any]) -> Callable[[Callable[..., Any]], Any]:
        def wrapper(*func_args: Any, **func_kwargs: Any) -> Any:
            self.__enter__()
            try:
                return func(*func_args, **func_kwargs)
            finally:
                self.__exit__()

        return wrapper


class CellsUpdaterProxy(ReadableProxy[CellUpdater, CellUpdaterList]):
    """Proxy that forwards attribute access to the ``updater`` field of each Cell.

    It returns a CellUpdater or CellUpdaterList object so that the user can call it
    directly, use it as a context manager or a decorator through the `run` method.

    See Also
    --------
    manim_grid.proxies.mobs_proxy.TagsProxy,
    manim_grid.proxies.mobs_proxy.MobsProxy,
    manim_grid.proxies.olds_proxy.OldsProxy

    """

    _attr = "updater"
    _bulk_container = CellUpdaterList
