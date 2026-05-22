from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, cast

import manim as m
from blinker import signal

from .base import ReadableProxy
from .config_proxy import Config

if TYPE_CHECKING:
    from manim_grid.grid import Cell


class CellUpdaterBase(ABC):
    """Define common logic for CellUpdater and CellUpdaterList."""

    @abstractmethod
    def __iter__(self) -> Iterator["CellUpdater"]: ...

    def __call__(self, keys: Iterable[str] = (), **overrides: Any) -> None:
        """Allow direct call: `grid.update_cells[...]()`.

        If neither `keys` nor `overrides` are passed, all the cell Config keys are
        applied in their order of priority.

        Parameters
        ----------
        keys
            Acts as a filter. Only the passed config keys will be applied, in the order
            they are passed and with the values as defined in the cell Config.
        **overrides
            Allows overriding config key values only for this call. It is possible to
            pass keys that are not defined in the cell Config. In that case, a signal
            will be sent for each of those keys as well.
        """
        for cell_updater in self:
            merged = cell_updater._merge_config(keys, **overrides)
            cell_updater._update(cell_updater, **merged)

    def run(self, keys: Iterable[str] = (), **overrides: Any) -> "_CMDecoWrapper":
        """Use as a context manager or decorator.

        If neither `keys` nor `overrides` are passed, all the cell Config keys are
        applied in their order of priority.

        Parameters
        ----------
        keys
            Acts as a filter. Only the passed config keys will be applied, in the order
            they are passed and with the values as defined in the cell Config.
        **overrides
            Allows overriding config key values only for this call. It is possible to
            pass keys that are not defined in the cell Config. In that case, a signal
            will be sent for each of those keys as well.
        """
        return _CMDecoWrapper(self, keys, **overrides)

    def _attach_updaters(self, keys: Iterable[str], **overrides: Any) -> None:
        for cell_updater in self:
            merged = cell_updater._merge_config(keys, **overrides)
            cell_updater._updater = partial(cell_updater._update, **merged)
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

    def _merge_config(
        self, keys: Iterable[str] = (), **overrides: Any
    ) -> Config | dict[str, Any]:
        """Determine the config keys to update, their order and their values."""
        config = self._owner.config.sort_by_priority()
        if not keys and not overrides:
            return config
        merged = config if not keys else {}
        for key in keys:
            try:
                merged[key] = config[key]
            except KeyError as e:
                raise KeyError(
                    f"{self._owner} does not have {key!r} config key."
                ) from e
        merged.update(overrides)
        return merged

    def _update(self, cell_updater: m.Mobject, **merged: Any) -> None:
        cell = cast("Cell", cell_updater._owner)
        for key, value in merged.items():
            signal("cell_updating").send(
                key, key=key, value=value, grid=cell._grid, cell=cell
            )


class CellUpdaterList(list[CellUpdater], CellUpdaterBase): ...


class _CMDecoWrapper:
    def __init__(
        self, parent: CellUpdaterBase, keys: Iterable[str], **overrides: Any
    ) -> None:
        self._parent = parent
        self._keys = keys
        self._overrides = overrides

    def __enter__(self) -> None:
        self._parent._attach_updaters(self._keys, **self._overrides)

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
