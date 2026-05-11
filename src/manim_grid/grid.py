from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar, Literal, Self, TypedDict, cast

import manim as m
import numpy as np
from blinker import signal
from manim.typing import Vector3DLike
from manim_utils import Stencil, get_bounds

from manim_grid.exceptions import (
    GridError,
    GridFrameError,
    GridLabelError,
    GridShapeError,
    GridStencilError,
)
from manim_grid.helpers import TrackedLazyAnimation
from manim_grid.labels import LabelMapper
from manim_grid.proxies.config_proxy import Config, ConfigProxy
from manim_grid.proxies.mobs_proxy import MobsProxy
from manim_grid.proxies.olds_proxy import OldsProxy
from manim_grid.proxies.rects_proxy import RectsProxy
from manim_grid.proxies.tags_proxy import Tags, TagsProxy


class EmptyMobject(m.VMobject):
    """Serve as a placeholder mobject in empty cells."""


class CellConfig(TypedDict):
    align: str | Vector3DLike
    mode: str


@dataclass
class Cell:
    """A single grid cell.

    Parameters
    ----------
    _grid
        The Grid object the cell belongs to.
    rect
        The rectangle that defines the cell’s geometric boundary.
    row_index
        The row index for that cell.
    col_index
        The column index for that cell.
    mob
        The *current* Mobject inside the cell. By default a placeholder
        :class:`EmptyMobject` instance is used so that the attribute always exists.
    old
        The *previous* object that occupied the cell. It is useful for transition
        effects (FadeOut, Transform, etc.). Also initialised with an ``EmptyMobject``.
    tags
        A :class:`proxy.tags_proxy.Tags` instance for user-defined metadata. The core
        library does not interpret this data; it is merely attached to the cell as a
        user convenience.

    """

    default_config: ClassVar[CellConfig] = {"align": m.ORIGIN, "mode": "NONE"}

    _grid: "Grid" = field(repr=False)
    rect: m.Rectangle = field(repr=False)
    row_index: int
    col_index: int
    mob: m.Mobject = field(default_factory=EmptyMobject, repr=False)
    old: m.Mobject = field(default_factory=EmptyMobject, repr=False)
    tags: Tags = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._grid.add(self.rect.set_opacity(0), self.old, self.mob)
        self.mob.name = f"Mob(EmptyMobject)[{self.row_index}, {self.col_index}]"
        self.old.name = f"Old(EmptyMobject)[{self.row_index}, {self.col_index}]"
        self.tags = Tags(owner=self)
        self.config = Config(owner=self, **self.default_config)

    def insert_mob(
        self,
        mob: m.Mobject,
        align: Vector3DLike | None,
        margin: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Insert a new mobject in the cell.

        This method performs three steps:

        1. Store the existing ``mob`` in ``self.old``.
        2. Assign the supplied ``mob`` to ``self.mob``.
        3. Position the new object inside ``self.rect`` using manim’s
           ``move_to``/``shift`` methods.

        Parameters
        ----------
        mob
            The new Mobject to place inside the cell.
        align
            A 3D vector that specifies which edge of ``self.rect`` the object should
            align to (e.g. ``m.UP``, ``m.DOWN``, ...). If `None`, the previously set
            vector for that cell is used.
        margin
            A three-component numpy array (``float64``) that offsets the object *away*
            from the aligned edge.
        """
        self.old = self.mob
        self.old.name = (
            f"Old({type(self.old).__name__})[{self.row_index}, {self.col_index}]"
        )
        self.mob = mob
        self.mob.name = (
            f"Mob({type(self.mob).__name__})[{self.row_index}, {self.col_index}]"
        )
        if align is not None:
            self.config["align"] = align

        alignment = self.config["align"]
        self.mob.move_to(self.rect, aligned_edge=alignment).shift(-alignment * margin)
        signal("mob_inserted").send(self, grid=self._grid)


class Grid(m.Group):
    """Provide a rectangular lattice of :class:`Cell` objects.

    The grid is responsible for:

    * creating the underlying ``np.ndarray`` of ``Cell`` instances,
    * arranging the rectangle placeholders in a Manim ``VGroup``,
    * adding a ``stencil`` in the form of a :class:`manim_utils.Stencil` object
      if at least one of ``num_visible_rows`` or ``num_visible_cols`` is specified.
    * exposing convenient proxy objects (``mobs``, ``olds``, ...) that forward
      attribute access to the underlying cells.

    Parameters
    ----------
    row_heights
        Sequence of heights (in munits) for each row. The length of this sequence
        determines the number of rows.
    col_widths
        Sequence of widths (in munits) for each column. The length of this sequence
        determines the number of columns.
    buff
        Spacing between cells. Either a scalar (applied to both axes) or a
        ``(horizontal, vertical)`` tuple.
    margin
        Global margin used when inserting a ``Mobject`` (passed to
        :meth:`Cell.insert_mob`). Accepts the same scalar/tuple convention as
        ``buff``.
    row_labels
        Optional sequence of strings that label the rows.
    col_labels
        Optional sequence of strings that label the columns.
    num_visible_rows
        The number of rows that should be visible. A :class:`manim_utils.Stencil`
        will be used to cover the hidden rows. This stencil is accessible through
        the attribute `grid.stencil`. If none of `num_visible_rows` and
        `num_visible_cols` is defined, the stencil will not be created.
    num_visible_cols
        Similar to `num_visible_rows` for columns.
    **kwargs
        Additional keyword arguments forwarded to the base ``Group``.

    Attributes
    ----------
    lattice
        The ``VGroup`` containing the Rectangle objects defining each cell boundary.
        Useful when acting on all rectangles at once:
        `grid.lattice.set_fill(WHITE, opacity=1)`
    rects
        A proxy giving access to the same Rectangles as a numpy array for greater
        control. For instance, targeting only the first column:
        `grid.rects[:, 0].set_fill(WHITE, opacity=1)`
    mobs
        A proxy giving access to the ``mob`` attribute of each cell. Supports
        read and write operations through ``__getitem__`` and ``__setitem__``.
    olds
        A proxy giving access to the ``old`` attribute of each cell. Supports
        read-only operation through ``__getitem__``.
    tags
        A proxy giving access to user defined key/value tags. Allows attaching
        metadata to cells. See :class:`manim_grid.proxies.tags_proxy.TagsProxy` for
        detailed instructions.
    gtags
        The `Tags` instance attached to the grid itself.
    config
        A proxy giving access to the cells configuration dictionary.

    """

    def __init__(
        self,
        row_heights: Sequence[float],
        col_widths: Sequence[float],
        *,
        buff: float | tuple[float, float] = 0.0,
        margin: float | tuple[float, float] = 0.1,
        row_labels: Sequence[str] = (),
        col_labels: Sequence[str] = (),
        num_visible_rows: int | None = None,
        num_visible_cols: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._stencil: Stencil | None = None
        self._frame: m.Difference | None = None
        super().__init__(**kwargs)

        num_rows, num_cols = len(row_heights), len(col_widths)
        self._row_heights = list(row_heights)
        self._col_widths = list(col_widths)
        self._buff = self._normalize_buff(buff)
        self._margin = self._normalize_margin(margin)

        self._row_labels = self._prepare_labels(row_labels, num_rows)
        self._col_labels = self._prepare_labels(col_labels, num_cols)
        self._label_mapper = LabelMapper(self._row_labels, self._col_labels)

        self.cells, self.lattice = self._prepare_grid(
            num_rows, num_cols, row_heights, col_widths, self._buff
        )

        self._num_visible_rows = num_visible_rows or num_rows
        self._num_visible_cols = num_visible_cols or num_cols
        self._first_visible_row = 0
        self._first_visible_col = 0

        self.rects = RectsProxy(self)
        self.mobs = MobsProxy(self, margin=self._margin)
        self.olds = OldsProxy(self)
        self.tags = TagsProxy(self)
        self.config = ConfigProxy(self)

        self.gtags = Tags(owner=self)

        if num_visible_rows is not None or num_visible_cols is not None:
            self._stencil = self._create_stencil()
            self.viewport = self._stencil.clip
            super().add(self._stencil)
        else:
            # some manim methods/classes (e.g. Difference) don't work with VGroup,
            # so we surround the lattice
            self.viewport = m.SurroundingRectangle(
                self.lattice, name="viewport", buff=0
            ).set_stroke(opacity=0)
            super().add(self.viewport)

    @classmethod
    def fullscreen(cls, num_rows: int, num_cols: int, **kwargs: Any) -> "Grid":
        """Alternative constructor to build a fullscreen Grid.

        The generated Grid will cover the whole scene frame and will have uniform row
        heights and column widths.


        Parameters
        ----------
        num_rows
            The desired number of rows.
        num_cols
            The desired number of columns.
        **kwargs
            Keyword arguments forwarded to `Grid.__init__`.
        """
        num_rows = int(num_rows)
        num_cols = int(num_cols)
        if num_rows < 1 or num_cols < 1:
            raise GridShapeError("A Grid should have at least 1 row and 1 column.")

        frame_h = m.config.frame_height
        frame_w = m.config.frame_width
        buff_tuple = (
            cls._normalize_buff(kwargs["buff"]) if "buff" in kwargs else (0.0, 0.0)
        )

        if buff_tuple[1] * (num_rows - 1) > frame_h:
            raise GridShapeError(
                f"The provided vertical buffer ({buff_tuple[1]}) is too large to fit "
                f"on the screen with {num_rows} rows. "
                f"It should be at most {int(frame_h / (num_rows - 1) * 100) / 100}."
            )
        if buff_tuple[0] * (num_cols - 1) > frame_w:
            raise GridShapeError(
                f"The provided horizontal buffer ({buff_tuple[0]}) is too large to fit "
                f"on the screen with {num_cols} columns. "
                f"It should be at most {int(frame_w / (num_cols - 1) * 100) / 100}."
            )

        row_height = (frame_h - (num_rows - 1) * buff_tuple[1]) / num_rows
        col_width = (frame_w - (num_cols - 1) * buff_tuple[0]) / num_cols

        grid = Grid([row_height] * num_rows, [col_width] * num_cols, **kwargs)
        return grid

    @staticmethod
    def _normalize_buff(buff: float | tuple[float, float]) -> tuple[float, float]:
        """Convert ``buff`` to a 2-tuple ``(horizontal, vertical)``.

        Returns
        -------
        tuple[float, float]
            ``(horizontal_spacing, vertical_spacing)``.

        Raises
        ------
        TypeError
            If *buff* cannot be converted to a 2-tuple of floats.
        """
        if isinstance(buff, (int, float)):
            return (float(buff), float(buff))
        elif isinstance(buff, tuple):
            if not all(isinstance(b, (int, float)) for b in buff):
                raise TypeError("Grid buffer should be a numeric value.")
            return (float(buff[0]), float(buff[1]))
        raise TypeError(
            "Grid buffer should be a numeric value or a 2-tuple of numeric values."
        )

    @staticmethod
    def _normalize_margin(
        margin: float | tuple[float, float],
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]:
        """Return a three-component ``ndarray`` suitable for ``Cell.insert_mob``.

        Returns
        -------
        ndarray
            A 3-component numpy array describing the margin to apply for each dimension.
            The grid lives in the XY-plane, so the Z component is always ``0.0``.

        Raises
        ------
        TypeError
            If *margin* cannot be converted to the desired output.
        """
        if isinstance(margin, (int, float)):
            return np.array([margin, margin, 0.0], dtype=np.float64)
        elif isinstance(margin, tuple):
            if not all(isinstance(m, (int, float)) for m in margin):
                raise TypeError("Grid margin should be a numeric value.")
            return np.array([margin[0], margin[1], 0.0], dtype=np.float64)
        raise TypeError(
            "Grid margin should be a numeric value or a 2-tuple of numeric values."
        )

    @staticmethod
    def _prepare_labels(labels: Sequence[str], num: int) -> dict[str, int]:
        """Map a sequence of labels to integer indices.

        Parameters
        ----------
        labels
            User-provided label sequence. Must be either empty or have length exactly
            ``num``.
        num
            Expected number of rows or columns.

        Returns
        -------
        dict[str, int]
            Mapping from ``label`` to ``index`` where ``index`` is zero-based.

        Raises
        ------
        ValueError
            If a non-empty *labels* sequence does not contain exactly ``num`` elements.
        """
        if labels == ():
            return {}

        nums = range(num)
        if num != len(labels):
            raise ValueError(
                "The number of labels should match the number of rows/columns. "
                f"({len(labels)} != {num})."
            )
        labels_dict: dict[str, int] = dict(zip(labels, nums, strict=True))
        return labels_dict

    def row_labels(self, **kwargs: Any) -> list[m.Text]:
        """Return the row labels as a list of Text Mobjects.

        This is a convenience method meant to easily add the labels to the grid.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to the `Text` constructor.

        Raises
        ------
        GridLableError
            If the Grid does not have row labels.

        """
        if not self._row_labels:
            raise GridLabelError("This Grid does not have row labels defined.")

        return [m.Text(label, **kwargs) for label in self._row_labels]

    def col_labels(self, **kwargs: Any) -> list[m.Text]:
        """Return the column labels as a list of Text Mobjects.

        This is a convenience method meant to easily add the labels to the grid.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to the `Text` constructor.

        Raises
        ------
        GridLableError
            If the Grid does not have column labels.

        """
        if not self._col_labels:
            raise GridLabelError("This Grid does not have column labels defined.")

        return [m.Text(label, **kwargs) for label in self._col_labels]

    def row_numbers(
        self, start: int = 1, stop: int | None = None, step: int = 1, **kwargs: Any
    ) -> list[m.Text]:
        """Return the row numbers as a list of Text Mobjects.

        This is a convenience method meant to easily add the numbers to the grid.

        Parameters
        ----------
        start
            An integer number specifying at which number to start.
        stop
            An optional integer number specifying at which number to stop.
            Defaults to the number that would be attributed to the last row starting
            from `start` with a step `step`.
        step
            An integer number specifying the incrementation.
        kwargs
            Keyword arguments passed to the `Text` constructor.

        """
        num_rows = len(self._row_heights)
        if stop is None:
            stop = start + num_rows * step
        return [m.Text(str(num), **kwargs) for num in range(start, stop, step)]

    def col_numbers(
        self, start: int = 1, stop: int | None = None, step: int = 1, **kwargs: Any
    ) -> list[m.Text]:
        """Return the column numbers as a list of Text Mobjects.

        This is a convenience method meant to easily add the numbers to the grid.

        Parameters
        ----------
        start
            An integer number specifying at which number to start.
        stop
            An optional integer number specifying at which number to stop.
            Defaults to the number that would be attributed to the last column starting
            from `start` with a step `step`.
        step
            An integer number specifying the incrementation.
        kwargs
            Keyword arguments passed to the `Text` constructor.

        """
        num_cols = len(self._col_widths)
        if stop is None:
            stop = start + num_cols * step
        return [m.Text(str(num), **kwargs) for num in range(start, stop, step)]

    def _prepare_grid(
        self,
        num_rows: int,
        num_cols: int,
        row_heights: Sequence[float],
        col_widths: Sequence[float],
        buff: tuple[float, float],
    ) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.object_]], m.VGroup]:
        """Create the internal ``Cell`` matrix and the lattice ``VGroup``.

        Parameters
        ----------
        num_rows, num_cols
            Dimensions of the grid.
        row_heights, col_widths
            Sequences defining the size of each row/column.
        buff
            ``(horizontal, vertical)`` spacing between cells.

        Returns
        -------
        tuple
            ``(cells, vgroup)`` where ``cells`` is the ``ndarray`` of :class:`Cell`
            objects and ``vgroup`` is the manim ``VGroup`` that holds the rectangles.
        """
        cells = np.empty((num_rows, num_cols), dtype=object)

        for i, row_h in enumerate(row_heights):
            for j, col_w in enumerate(col_widths):
                rect = m.Rectangle(
                    name=f"Rect[{i}, {j}]",
                    height=row_h,
                    width=col_w,
                )
                cells[i, j] = Cell(self, row_index=i, col_index=j, rect=rect)

        lattice = m.VGroup(cell.rect for cell in cells.ravel())
        lattice.arrange_in_grid(
            rows=num_rows,
            cols=num_cols,
            buff=buff,
            aligned_edge=m.UP,
        )
        return cells, lattice

    def add(self, *mobjects: m.Mobject) -> Self:
        """Add mobjects as submobjects.

        This overriden method makes sure the right order of submobjects is preserved.
        The stencil should cover the newly added mobjects and the frame should cover the
        stencil.

        """
        super().add(*mobjects)
        for mob in mobjects:
            signal("mob_added").send(mob, grid=self)

        if self._stencil is not None:
            super().add(self.stencil)
        if self._frame is not None:
            super().add(self.frame)
        return self

    def remove(self, *mobjects: m.Mobject) -> Self:
        """Remove submobjects."""
        super().remove(*mobjects)
        for mob in mobjects:
            signal("mob_removed").send(mob, grid=self)
        return self

    @property
    def stencil(self) -> Stencil:
        """A property giving access to the stencil if it exists.

        Returns
        -------
        The Stencil object if it exists.

        Raises
        ------
        GridStencilError if it does not exist.
        """
        if self._stencil is None:
            raise GridStencilError(
                "This Grid does not have a stencil. Define `num_visible_rows` "
                "and/or `num_visible_cols` to generate one."
            )
        return self._stencil

    def _create_stencil(self) -> Stencil:
        """Create the stencil to hide cells that should not be visible."""
        viewport = self._compute_viewport(m.Rectangle())
        return Stencil(
            # add viewport to wrapped to prevent weird artifacts when scrolling
            # past the last row/line
            clip=viewport,
            wrapped=m.VGroup(self.lattice, viewport),
            name="Stencil",
        ).set_stroke(opacity=0)

    def _compute_viewport(
        self,
        viewport: m.Mobject,
        predicate: Callable[..., bool] = lambda: True,
        **kwargs: Any,
    ) -> m.VMobject:
        """(Re)Compute the viewport.

        Can be used as an updater using a partial or lambda to set the extra parameters.

        Parameters
        ----------
        viewport
            The viewport (=stencil.clip) that will be computed/updated.
        predicate
            A callable taking any number of keyword arguments and returning a boolean.
            The code will be executed only when this predicate returns `True`.
        **kwargs
            Keyword arguments passed to the predicate.

        Returns
        -------
        The viewport mobject, modified or not.

        """
        if predicate(**kwargs):
            to_row = self._first_visible_row + self._num_visible_rows
            to_col = self._first_visible_col + self._num_visible_cols

            # take viewport and rects stroke widths into account (in 1/100 munits)
            visible_rects = self.rects[
                self._first_visible_row : to_row, self._first_visible_col : to_col
            ]

            w, h, _, center = get_bounds(
                visible_rects, as_len=True, include_stroke=True
            )
            target = m.Rectangle(width=w, height=h).move_to(center)
            viewport.match_points(target)

        return cast(m.VMobject, viewport)

    @contextmanager
    def update_viewport(
        self,
        predicate: Callable[..., bool] = lambda: True,
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        """Recompute the viewport to encompass the visible rows/cols while active.

        This is meant to be used as a context manager. While active, the viewport will
        be updated. Useful in insertion methods for example when the inserted
        row/column has a size that differs from the last visible row/column.

        Because updaters can be expensive:
         - this context manager attaches the updater while entering and detaches it
           when exiting. This allows to target a specific piece of code easily.
         - a predicate function can be passed to target even more precisely when the
           updater should run and when it shouldn't. This is also very useful to start
           the viewport updating only when a given condition is met.

        Parameters
        ----------
        predicate
            A callable taking any number of arguments and keyword arguments and
            returning a boolean. The updater code will be executed only when this
            predicate returns `True`.
        **kwargs
            Keyword arguments passed to the predicate.

        """
        updater_func = partial(self._compute_viewport, predicate=predicate, **kwargs)
        self.viewport.add_updater(updater_func)

        if self._frame is not None:
            self.frame.add_updater(self._update_frame)
        try:
            yield
        finally:
            # update viewport (and stencil to cover it if it expands) for static frames
            updater_func(viewport=self.viewport)
            if self._stencil is not None:
                self.stencil.update()

            self.viewport.remove_updater(updater_func)
            if self._frame is not None:
                self.frame.remove_updater(self._update_frame)

    @property
    def frame(self) -> m.Difference:
        """Return the frame VMobject.

        Raises
        ------
        GridFrameError
            If the frame does not exist.
        """
        if self._frame is None:
            raise GridFrameError(
                "This Grid does not have a frame. Set it via `grid.frame = ...` first."
            )
        return self._frame

    @frame.setter
    def frame(self, vmobject: m.VMobject | None) -> None:
        """Set the frame.

        Providing ``None`` will remove any previous frame.
        Providing a VMobject will compute the difference between the provided VMobject
        and the grid viewport. It is the user responsibility to position the frame where
        it should be.
        """
        with suppress(GridFrameError):
            self.remove(self.frame)

        if vmobject is None:
            self._frame = None

        else:
            self._frame_vmob = vmobject
            self._frame_top_margin = abs(
                self.viewport.get_y(m.UP) - vmobject.get_y(m.UP)
            )
            self._frame_bottom_margin = abs(
                self.viewport.get_y(m.DOWN) - vmobject.get_y(m.DOWN)
            )
            self._frame_right_margin = abs(
                self.viewport.get_x(m.RIGHT) - vmobject.get_x(m.RIGHT)
            )
            self._frame_left_margin = abs(
                self.viewport.get_x(m.LEFT) - vmobject.get_x(m.LEFT)
            )
            self._frame = self._make_frame(vmobject)
            self.add(self._frame)

    def _update_frame(self, frame: m.Mobject) -> None:
        """Update the frame to the viewport size keeping margins fixed."""
        vp_stroke = self.viewport.get_stroke_width() / 100
        vmobject = self._frame_vmob.stretch_to_fit_height(
            self.viewport.height
            + self._frame_top_margin
            + self._frame_bottom_margin
            + vp_stroke
        )
        vmobject.stretch_to_fit_width(
            self.viewport.width
            + self._frame_right_margin
            + self._frame_left_margin
            + vp_stroke
        )

        vmobject.align_to(self.viewport, m.UP).shift(
            m.UP * (self._frame_top_margin + vp_stroke / 2)
        )
        vmobject.align_to(self.viewport, m.LEFT).shift(
            m.LEFT * (self._frame_left_margin + vp_stroke / 2)
        )

        frame = self._make_frame(vmobject)
        self.frame.set_points(frame.points)

    def _make_frame(self, vmobject: m.VMobject) -> m.Difference:
        """Build the frame Difference from a vmobject."""
        vp_stroke = self.viewport.get_stroke_width() / 100
        clip = m.SurroundingRectangle(self.viewport, buff=vp_stroke / 2)
        frame = m.Difference(vmobject, clip, name="Frame").match_style(vmobject)
        return frame

    @contextmanager
    def keep_viewport_static(self) -> Generator[None, None, None]:
        """Keep the viewport in place while this context manager is active.

        The "viewport" here is meant in a visual sense. It is not only composed by the
        `grid.viewport` Rectangle, but also the `Grid.frame` Difference.
        These submojects usually stay in sync with the grid (e.g. when we shift the
        grid, we usually want the visual viewport to be shifted as well). However, we
        sometimes need to transform the Grid without transforming the visual viewport
        with it (e.g. when scrolling). This context manager simply removes the frame
        from the grid submobjects and set the viewport (the stencil clip) to be static
        while entering and restores everything when exiting.

        Raises
        ------
        GridError
            When the Grid does not have a stencil. It does not make much sense to keep
            the viewport static when it encompases the whole Grid.
        """
        if self._stencil is None:
            raise GridError(
                "`keep_viewport_static` can only be used on a Grid with a stencil."
            )
        self.stencil.is_clip_static = True
        if self._frame is not None:
            self.remove(self.frame)
        if self._stencil is None:
            self.remove(self.viewport)
        try:
            yield
        finally:
            self.stencil.is_clip_static = False
            if self._frame is not None:
                self.add(self.frame)
            if self._stencil is None:
                self.add(self.viewport)

    @property
    def has_uniform_rows(self) -> bool:
        """Return ``True`` iff all the grid rows have the same height."""
        return len(set(self._row_heights)) == 1

    @property
    def has_uniform_cols(self) -> bool:
        """Return ``True`` iff all the grid cols have the same width."""
        return len(set(self._col_widths)) == 1

    def scroll(self, direction: Vector3DLike, step: int) -> Self:
        """Scroll the grid horizontally and/or vertically a given number of cells.

        This method scrolls the Grid an integer number of rows/columns. The Grid must
        have uniform cell size in the direction of the scrolling.

        Parameters
        ----------
        direction
            The direction in which to scroll. Any manim `Vector3DLike` will do.
        step
            The number of cells to scroll for.

        Returns
        -------
        Self
            The grid itself. This allows to animate the scrolling and chain animations:
            `self.play(grid.animate.scroll(DOWN, 3).set_color(RED))`

        Raises
        ------
        GridStencilError
            If no stencil is defined.
        GridShapeError
            If the grid does not have uniform row heights for vertical scrolling or
            uniform column widths for horizontal scrolling.

        See Also
        --------
        :meth:`free_scroll`
        """
        if self._stencil is None:
            raise GridStencilError(
                "A grid without a stencil cannot be scrolled. "
                "Define `num_visible_rows` or `num_visible_cols` or both."
            )

        if direction[0] != 0:
            if not self.has_uniform_cols:
                raise GridShapeError(
                    "In order to scroll horizontally, the grid must have "
                    "uniform column widths."
                )
            self._first_visible_col += int(direction[0] * step)

        if direction[1] != 0:
            if not self.has_uniform_rows:
                raise GridShapeError(
                    "In order to scroll vertically, the grid must have uniform "
                    "row heights."
                )
            self._first_visible_row -= int(direction[1] * step)

        offset = self._compute_scroll_offset(direction, step)
        # viewport and grid.frame should not be shifted
        with self.keep_viewport_static():
            self.shift(offset)

        # make sure the stencil is recomputed even for a static frame
        self.stencil.update()
        return self

    def _compute_scroll_offset(
        self, direction: Vector3DLike, step: int
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Compute the vector by which to shift the grid.

        Parameters
        ----------
        direction
            The direction in which to scroll. Any manim `Vector3DLike` will do.
        step
            The number of cells to scroll for.

        Returns
        -------
        ndarray
            Each component encodes the amount by which to shift the grid based on the
            provided parameters, the row height, column width, horizontal and vertical
            buffers.
        """
        one_cell_offset = np.array(
            [self.lattice[0].width, self.lattice[0].height, 0.0]
        ) + np.array([*self._buff, 0.0])

        offset = (
            one_cell_offset
            * -1  # Scrolling UP means shifting DOWN.
            * direction
            * step
        )
        return np.array(offset)

    def free_scroll(self, direction: Vector3DLike, munits: float = 1) -> Self:
        """Scroll the grid horizontally and/or vertically in a free way.

        Unlike :meth:`scroll`, the cell size does not have to be uniform in the
        direction of the scrolling. The drawback is that this will not update
        :attr:`_first_visible_row` and :attr:`_first_visible_col`, meaning that updating
        the viewport later could cause unwanted jumps.

        Parameters
        ----------
        direction
            The direction in which to scroll. Any manim `Vector3DLike` will do.
        munits
            The number of munits to scroll for.

        Raises
        ------
        GridStencilError
            If no stencil is defined.

        Returns
        -------
        Self
            The grid itself. This allows to animate the scrolling and chain animations:
            `self.play(grid.animate.scroll(DOWN, 3).set_color(RED))`

        See Also
        --------
        :meth:`scroll`
        """
        if self._stencil is None:
            raise GridStencilError(
                "A grid without a stencil cannot be scrolled. "
                "Define `num_visible_rows` or `num_visible_cols` or both."
            )

        # viewport and grid.frame should not be shifted
        with self.keep_viewport_static():
            self.shift(np.array(direction) * -munits)

        # make sure the stencil is recomputed even for a static frame
        self.stencil.update()
        return self

    @contextmanager
    def insert_row(
        self,
        row_index: int | str,
        *,
        height: float | None = None,
        label: str | None = None,
        shift_tags: bool = False,
    ) -> Generator[tuple[TrackedLazyAnimation, m.VDict, m.ValueTracker], None, None]:
        """Insert a new row in the Grid.

        The Grid geometry will not be changed and cells identity is preserved after
        insertion. The last row mobjects will be removed from `grid.mobs` as well as
        `grid.submobjects`. To avoid this visually, extra empty rows must be
        pre-allocated.

        This method acts as a context manager providing an opportunity to change
        the Grid (e.g. style the new row or change displayed string labels or row
        numbers...) before the insertion takes place, and to animate this insertion.
        Inside the context manager, the Grid is already in its post-insertion internal
        state (e.g. the new row is accessible via `grid.mobs[row_index]`). When exiting
        the context manager, the visual aspect of the insertion will take place.

        Parameters
        ----------
        row_index
            The integer index (or its string label) where the new row should be
            inserted.
        height
            The height in munits of the new row. It can be omitted if the Grid has
            uniform row heights and will default to that uniform height.
        label
            The string label to attribute to the newly inserted row. Must be provided
            if and only if the other rows already have string labels.
        shift_tags
            Whether or not the tags should be shifted to the next row cell during
            insertion. If you consider tags to be part of the content (i.e. they
            describe the mobs inside each cell) set to `True`, else (i.e. they are
            attached to the Cells themselves and describe the position), keep the
            `False` default.

        Yields
        ------
        tuple[Animation, VDict, ValueTracker]
            The first element in the yielded tuple is the shift animation for the rows
            below the inserted one. It can be played directly. If not played, an instant
            shift will happen when exiting the context manager.
            The second element is a VDict containing the last row mobjects
            (keys: `mobs`, `olds` and `rects`). It can be used to animate the last row
            removal (e.g. FadeOut) or style it. These mobjects will be removed from the
            grid when exiting the context manager.
            The third element is a ValueTracker tracking the advancement of the shift
            animation. Can be useful in conjunction with :meth:`update_viewport` for
            instance to precisely time the start of the viewport animation.

        Raises
        ------
        GridShapeError
            If the row height is not passed for a non-uniform Grid.
        GridLabelError
            If a `label` for the new row is passed when the Grid does not have labels
            defined, or if the `label` is not passed when it should be.

        Examples
        --------
        >>> # simplest form: no pre-styling, no animation (automatic instant shift)
        >>> with grid.insert_row(3): pass

        >>> # with pre-styling and animation
        >>> with grid.insert_row(3, label="new_row") as (anim, last_row, tracker):
        >>>     # the grid is already in the post insertion state internally
        >>>     grid.rects["new_row"].set_stroke(opacity=1)
        >>>     self.play(FadeOut(last_row))
        >>>     self.play(anim, run_time=2)

        """
        if isinstance(row_index, str):
            row_index = self._row_labels[row_index]

        num_rows = len(self._row_heights)
        num_cols = len(self._col_widths)

        # update _row_heights
        if height is None:
            if not self.has_uniform_rows:
                raise GridShapeError(
                    "This Grid does not have uniform row heights. "
                    "You must provide the height for the inserted row."
                )
            else:
                height = self._row_heights[0]

        self._row_heights.insert(row_index, height)
        self._row_heights.pop()

        # update _row_labels and LabelMapper
        if label is None and self._row_labels:
            raise GridLabelError(
                "You must provide a string label for the inserted row."
            )

        if label is not None:
            if not self._row_labels:
                raise GridLabelError(
                    "This Grid does not have row labels defined. You cannot define one "
                    "for the inserted row."
                )
            else:
                labels = list(self._row_labels.keys())
                labels.insert(row_index, label)
                labels.pop()
                self._row_labels = self._prepare_labels(labels, num_rows)

        self._label_mapper = LabelMapper(self._row_labels, self._col_labels)

        # shift references
        # NOTE: self.cells[row_index + 1 :, :] = self.cells[row_index:-1, :]
        # would breack Cells identity => shift member mobjects instead

        d: Mapping[Hashable, m.VMobject] = {
            "mobs": self.mobs[-1],
            "olds": self.olds[-1],
            "rects": self.rects[-1],
        }
        last_row = m.VDict(d)

        attrs_to_shift = ["config", "mob", "old", "rect"]
        if shift_tags:
            attrs_to_shift.append("tags")

        for row in range(num_rows - 2, row_index - 1, -1):
            for col in range(num_cols):
                for attr in attrs_to_shift:
                    value = getattr(self.cells[row, col], attr)
                    setattr(self.cells[row + 1, col], attr, value)

        # reset inserted row and update lattice
        # NOTE: we remove the last row rects from the lattice at the very end to make
        # sure the stencil covers the last row the whole time
        idx = num_cols * row_index
        for col_index, width in enumerate(self._col_widths):
            cell = self.cells[row_index, col_index]
            rect = (
                m.Rectangle(height=height, width=width)
                .set_opacity(0)
                .move_to(cell.rect, aligned_edge=m.UL)
            )
            cell.rect = rect
            cell.config = Config(owner=cell, **Cell.default_config)
            cell.mob = EmptyMobject()
            cell.old = EmptyMobject()
            if shift_tags:
                cell.tags = Tags(owner=cell)

            self.lattice.insert(idx, rect)
            self.add(rect)
            idx += 1

        # visually shift mobs/olds/rects
        # NOTE: since the user could add/remove mobjects inside the context manager, we
        # yield a LazyAnimation. The grp is defined now but built when the animation
        # is played.
        rows_to_shift = slice(row_index + 1, None)

        def grp_factory() -> m.VGroup:
            return m.VGroup(
                *self.mobs[rows_to_shift],
                *self.olds[rows_to_shift],
                *self.rects[rows_to_shift],
            )

        shift_vec = m.DOWN * (height + self._buff[1])

        def animation_factory(mob: m.Mobject) -> m.Animation:
            return m.ApplyMethod(mob.shift, shift_vec)

        alpha_tracker = m.ValueTracker()

        animation = TrackedLazyAnimation(
            alpha_tracker=alpha_tracker,
            animation_factory=animation_factory,
            mobject_factory=grp_factory,
        )

        try:
            yield (animation, last_row, alpha_tracker)

            signal("row_insertion_processed").send(
                self,
                row_index=row_index,
                height=height,
                label=label,
                animation=animation,
                tracker=alpha_tracker,
                shift_group_factory=grp_factory,
                shift_vec=shift_vec,
                last_row=last_row,
            )

        finally:
            for vgrp in last_row:
                self.remove(*vgrp)
            if animation.status == "not played":
                grp = grp_factory()
                grp.shift(shift_vec)
            self.lattice.remove(*self.lattice[-num_cols:])

            signal("row_insertion_displayed").send(
                self,
                row_index=row_index,
                height=height,
                label=label,
                animation=animation,
                tracker=alpha_tracker,
                shift_group=grp_factory(),
                shift_vec=shift_vec,
                last_row=last_row,
            )

    @contextmanager
    def insert_column(
        self,
        col_index: int | str,
        *,
        width: float | None = None,
        label: str | None = None,
        shift_tags: bool = False,
    ) -> Generator[tuple[TrackedLazyAnimation, m.VDict, m.ValueTracker], None, None]:
        """Insert a new column in the Grid.

        The Grid geometry will not be changed and cells identity is preserved after
        insertion. The last column mobjects will be removed from `grid.mobs` as well as
        `grid.submobjects`. To avoid this visually, extra empty columns must be
        pre-allocated.

        This method acts as a context manager providing an opportunity to
        change the Grid (e.g. style the new column or change displayed string labels or
        column numbers...) before the insertion takes place, and to animate this
        insertion.
        Inside the context manager, the Grid is already in its post-insertion internal
        state (e.g. the new column is accessible via `grid.mobs[:, col_index]`).
        When exiting the context manager, the visual aspect of the insertion will take
        place.

        Parameters
        ----------
        col_index
            The integer index (or its string label) where the new column should be
            inserted.
        width
            The width in munits of the new column. It can be omitted if the Grid has
            uniform column widths and will default to that uniform width.
        label
            The string label to attribute to the newly inserted column. Must be provided
            if and only if the other columns already have string labels.
        shift_tags
            Whether or not the tags should be shifted to the next column cell during
            insertion. If you consider tags to be part of the content (i.e. they
            describe the mobs inside each cell) set to `True`, else (i.e. they are
            attached to the Cells themselves and describe the position), keep the
            `False` default.

        Yields
        ------
        tuple[Animation, VDict, ValueTracker]
            The first element in the yielded tuple is the shift animation for the
            columns below the inserted one. It can be played directly. If not played,
            an instant shift will happen when exiting the context manager.
            The second element is a VDict containing the last column mobjects
            (keys: `mobs`, `olds` and `rects`). It can be used to animate the last col
            removal (e.g. FadeOut) or style it. These mobjects will be removed from the
            grid when exiting the context manager.
            The third element is a ValueTracker tracking the advancement of the shift
            animation. Can be useful in conjunction with :meth:`update_viewport` for
            instance to precisely time the start of the viewport animation.

        Raises
        ------
        GridShapeError
            If the column width is not passed for a non-uniform Grid.
        GridLabelError
            If a `label` for the new column is passed when the Grid does not have labels
            defined, or if the `label` is not passed when it should be.

        Examples
        --------
        >>> # simplest form: no pre-styling, no animation (automatic instant shift)
        >>> with grid.insert_column(3): pass

        >>> # with pre-styling and animation
        >>> with grid.insert_column(3, label="new_col") as (anim, last_col, tracker):
        >>>     # the grid is already in the post insertion state internally
        >>>     grid.rects[:, "new_col"].set_stroke(opacity=1)
        >>>     self.play(FadeOut(last_col))
        >>>     self.play(anim, run_time=2)

        """
        if isinstance(col_index, str):
            col_index = self._col_labels[col_index]

        num_rows = len(self._row_heights)
        num_cols = len(self._col_widths)

        # update _col_widths
        if width is None:
            if not self.has_uniform_cols:
                raise GridShapeError(
                    "This Grid does not have uniform column widths. "
                    "You must provide the width for the inserted column."
                )
            else:
                width = self._col_widths[0]

        self._col_widths.insert(col_index, width)
        self._col_widths.pop()

        # update _column_labels and LabelMapper
        if label is None and self._col_labels:
            raise GridLabelError(
                "You must provide a string label for the inserted column."
            )

        if label is not None:
            if not self._col_labels:
                raise GridLabelError(
                    "This Grid does not have column labels defined. You cannot define "
                    "one for the inserted column."
                )
            else:
                labels = list(self._col_labels.keys())
                labels.insert(col_index, label)
                labels.pop()
                self._col_labels = self._prepare_labels(labels, num_cols)

        self._label_mapper = LabelMapper(self._row_labels, self._col_labels)

        # shift references
        # NOTE: self.cells[:, col_index + 1 :] = self.cells[:, col_index:-1]
        # would breack Cells identity => shift member mobjects instead

        d: Mapping[Hashable, m.VMobject] = {
            "mobs": self.mobs[:, -1],
            "olds": self.olds[:, -1],
            "rects": self.rects[:, -1],
        }
        last_col = m.VDict(d)
        last_col_rects = self.rects[:, -1]

        attrs_to_shift = ["config", "mob", "old", "rect"]
        if shift_tags:
            attrs_to_shift.append("tags")

        for col in range(num_cols - 2, col_index - 1, -1):
            for row in range(num_rows):
                for attr in attrs_to_shift:
                    value = getattr(self.cells[row, col], attr)
                    setattr(self.cells[row, col + 1], attr, value)

        # reset inserted col and update lattice
        # NOTE: we remove the last col rects from the lattice at the very end to make
        # sure the stencil covers the last col the whole time
        for row_index, height in enumerate(self._row_heights):
            idx = (row_index * num_cols) + col_index + row_index
            cell = self.cells[row_index, col_index]
            rect = (
                m.Rectangle(height=height, width=width)
                .set_opacity(0)
                .move_to(cell.rect, aligned_edge=m.UL)
            )
            cell.rect = rect
            cell.config = Config(owner=cell, **Cell.default_config)
            cell.mob = EmptyMobject()
            cell.old = EmptyMobject()
            if shift_tags:
                cell.tags = Tags(owner=cell)

            self.lattice.insert(idx, rect)
            self.add(rect)
            idx += 1

        # visually shift mobs/olds/rects
        # NOTE: since the user could add/remove mobjects inside the context manager, we
        # yield a LazyAnimation. The grp is defined now but built when the animation
        # is played.
        cols_to_shift = slice(col_index + 1, None)

        def grp_factory() -> m.VGroup:
            return m.VGroup(
                *self.mobs[:, cols_to_shift],
                *self.olds[:, cols_to_shift],
                *self.rects[:, cols_to_shift],
            )

        shift_vec = m.RIGHT * (width + self._buff[0])

        def animation_factory(mob: m.Mobject) -> m.Animation:
            return m.ApplyMethod(mob.shift, shift_vec)

        alpha_tracker = m.ValueTracker()

        animation = TrackedLazyAnimation(
            alpha_tracker=alpha_tracker,
            animation_factory=animation_factory,
            mobject_factory=grp_factory,
        )

        try:
            yield (animation, last_col, alpha_tracker)

            signal("column_insertion_processed").send(
                self,
                col_index=col_index,
                width=width,
                label=label,
                animation=animation,
                tracker=alpha_tracker,
                shift_group_factory=grp_factory,
                shift_vec=shift_vec,
                last_col=last_col,
            )

        finally:
            for vgrp in last_col:
                self.remove(*vgrp)
            if animation.status == "not played":
                grp = grp_factory()
                grp.shift(shift_vec)
            self.lattice.remove(*last_col_rects)

            signal("column_insertion_displayed").send(
                self,
                col_index=col_index,
                width=width,
                label=label,
                animation=animation,
                tracker=alpha_tracker,
                shift_group=grp_factory(),
                shift_vec=shift_vec,
                last_col=last_col,
            )
