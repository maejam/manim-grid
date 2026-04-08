from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import manim as m
import numpy as np
from blinker import signal
from manim.typing import Vector3D, Vector3DLike
from manim_utils import Stencil

from manim_grid.exceptions import (
    GridError,
    GridFrameError,
    GridShapeError,
    GridStencilError,
)
from manim_grid.labels import LabelMapper
from manim_grid.proxies.mobs_proxy import MobsProxy
from manim_grid.proxies.olds_proxy import OldsProxy
from manim_grid.proxies.rects_proxy import RectsProxy
from manim_grid.proxies.tags_proxy import Tags, TagsProxy


class EmptyMobject(m.VMobject):
    """Serve as a placeholder mobject in empty cells."""


@dataclass
class Cell:
    """A single grid cell.

    Parameters
    ----------
    grid
        The Grid object the cell belongs to.
    rect
        The rectangle that defines the cell’s geometric boundary.
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

    _grid: "Grid" = field(repr=False)
    rect: m.Rectangle = field(repr=False)
    row_index: int
    col_index: int
    mob: m.Mobject = field(default_factory=EmptyMobject, repr=False)
    old: m.Mobject = field(default_factory=EmptyMobject, repr=False)
    tags: Tags = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._grid.add(self.rect.set_opacity(0), self.old, self.mob)
        self.tags = Tags(cell=self)
        self.alignment = m.ORIGIN

    def insert_mob(
        self,
        mob: m.Mobject,
        alignment: Vector3D,
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
        alignment
            A 3D vector that specifies which edge of ``self.rect`` the object should
            align to (e.g. ``m.UP``, ``m.DOWN``, ...).
        margin
            A three-component numpy array (``float64``) that offsets the object *away*
            from the aligned edge.
        """
        self.old = self.mob
        self.mob = mob
        self.alignment = alignment if alignment is not None else self.alignment
        self.mob.move_to(self.rect, aligned_edge=self.alignment).shift(
            -self.alignment * margin
        )
        signal("mob_inserted").send(self, grid=self._grid)


class Grid(m.Group):
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
            Optional sequence of strings that label the rows. If omitted, numeric
            strings (``"1"``, ``"2"``, ...) are generated automatically.
        col_labels
            Optional sequence of strings that label the columns. Same fallback behaviour
            as ``row_labels``.
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

        """
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

        if num_visible_rows is not None or num_visible_cols is not None:
            self._stencil = self._create_stencil(
                self._num_visible_rows, self._num_visible_cols
            )
            self.viewport = self._stencil.clip.copy()
            super().add(self._stencil)
        else:
            # some manim methods/classes (e.g. Difference) don't work with VGroup,
            # so we surround the lattice
            self.viewport = m.SurroundingRectangle(self.lattice, buff=0).set_stroke(
                opacity=0
            )
        super().add(self.viewport)

        self.rects = RectsProxy(self)
        self.mobs = MobsProxy(self, margin=self._margin)
        self.olds = OldsProxy(self)
        self.tags = TagsProxy(self)

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

        If *labels* is empty, numeric strings ``"1"``, ``"2"``, ... up to ``num`` are
        generated automatically.

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
            labels = tuple(map(str, range(1, num + 1)))

        nums = range(num)
        if len(nums) != len(labels):
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

        """
        return [m.Text(label, **kwargs) for label in self._row_labels]

    def col_labels(self, **kwargs: Any) -> list[m.Text]:
        """Return the column labels as a list of Text Mobjects.

        This is a convenience method meant to easily add the labels to the grid.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to the `Text` constructor.

        """
        return [m.Text(label, **kwargs) for label in self._col_labels]

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

        This overriden method performs 2 tasks:
        * it unpacks Groups and VGroups to add their submobjects instead of adding the
          groups themselves. This allows a more natural Grid workflow: notations like
          `grid.add(grid.mobs[:])` will add the leaf `mobs`; removing them later does
          not require to keep a reference to the Group returned by the MobsProxy.
          This is not recursive: only one level is unpacked.
        * it makes sure the right order of submobjects is preserved. The stencil should
          cover the newly added mobjects, the frame should cover the stencil and finally
          the viewport should be last.
        """
        for mob in mobjects:
            if isinstance(mob, (m.Group, m.VGroup)):
                super().add(*mob.submobjects)
            else:
                super().add(mob)

        if self._stencil is not None:
            super().add(self.stencil)
        if self._frame is not None:
            super().add(self.frame)
        if hasattr(self, "viewport"):
            super().add(self.viewport)
        return self

    def add_to_back(self, *mobjects: m.Mobject) -> Self:
        """Add all passed mobjects to the back of the submobjects.

        Similarly to `add`, unpacks (V)Groups to add individual submobjects.
        """
        for mob in mobjects:
            if isinstance(mob, (m.Group, m.VGroup)):
                super().add_to_back(*mob.submobjects)
            else:
                super().add_to_back(mob)
        return self

    def remove(self, *mobjects: m.Mobject) -> Self:
        """Remove submobjects.

        (V)Groups are unpacked to directly remove members submojects.
        """
        for mob in mobjects:
            if isinstance(mob, (m.Group, m.VGroup)):
                super().remove(*mob.submobjects)
            else:
                super().remove(mob)
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

    def _create_stencil(self, num_rows: int, num_cols: int) -> Stencil:
        """Create the stencil to hide cells that should not be visible."""
        visible_area = [
            cell.rect for cell in self.cells[:num_rows, :num_cols].flatten()
        ]
        clip = m.SurroundingRectangle(m.VGroup(visible_area), buff=0)
        return Stencil(clip=clip, wrapped=self.lattice).set_stroke(opacity=0)

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
        if vmobject is None:
            if self.frame is not None:
                self.remove(self.frame)
        else:
            self._frame = m.Difference(vmobject, self.viewport).match_style(vmobject)
            self.add(self._frame)

    @contextmanager
    def keep_viewport_static(self) -> Generator[None, None, None]:
        """Keep the viewport in place while this context manager is active.

        The "viewport" here is meant in a visual sense. It is not only composed by the
        `grid.viewport` Rectangle, but also the `Grid.frame` Difference and the
        `Grid.stencil.clip` Rectangle. All these submojects usually stay in sync with
        the grid (e.g. when we shift the grid, we usually want the visual viewport to be
        shifted as well). However, we sometimes need to transform the Grid without
        transforming the visual viewport with it (e.g. when scrolling). This context
        manager simply removes the listed submobjects from the grid while entering and
        adds them back when exiting.

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
        self.remove(self.viewport)
        yield
        self.stencil.is_clip_static = False
        if self._frame is not None:
            self.add(self.frame)
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

        if direction[0] != 0 and not self.has_uniform_cols:
            raise GridShapeError(
                "In order to scroll horizontally, the grid must have "
                "uniform column widths."
            )

        if direction[1] != 0 and not self.has_uniform_rows:
            raise GridShapeError(
                "In order to scroll vertically, the grid must have uniform row heights."
            )

        offset = self._compute_scroll_offset(direction, step)
        # stencil.clip and grid.frame should not be shifted
        with self.keep_viewport_static():
            self.shift(offset)

        # make sure the stencil is recomputed even with no further animation
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

    def free_scroll(self, direction: Vector3DLike, munits: float) -> Self:
        """Scroll the grid horizontally and/or vertically in a free way.

        Unlike :meth:`scroll`, the cell size does not have to be uniform in the
        direction of the scrolling.

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

        # stencil.clip and grid.frame should not be shifted
        with self.keep_viewport_static():
            self.shift(np.array(direction) * -munits)

        # make sure the stencil is recomputed even with no further animation
        self.stencil.update()
        return self
