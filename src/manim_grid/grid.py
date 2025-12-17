from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import manim as m
import numpy as np
from manim.typing import Vector3D, Vector3DLike

from manim_grid.exceptions import GridShapeError
from manim_grid.labels import LabelMapper
from manim_grid.proxies.mobs_proxy import MobsProxy
from manim_grid.proxies.olds_proxy import OldsProxy
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
        An class instance for user-defined metadata. The core library does not interpret
        this data; it is merely attached to the cell as a user convenience.
    """

    _grid: "Grid" = field(repr=False)
    rect: m.Rectangle = field(repr=False)
    mob: m.Mobject = field(default_factory=EmptyMobject)
    old: m.Mobject = field(default_factory=EmptyMobject)
    tags: Tags = field(default_factory=Tags)

    def __post_init__(self) -> None:
        # Add as submojects to the grid so that they move around with it.
        self._grid.add(self.rect, self.mob, self.old)

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
        self.mob.move_to(self.rect, aligned_edge=alignment).shift(-alignment * margin)
        self._grid.add(self.mob)


class Grid(m.Mobject):
    def __init__(
        self,
        row_heights: Sequence[float],
        col_widths: Sequence[float],
        *,
        buff: float | tuple[float, float] = 0.0,
        margin: float | tuple[float, float] = 0.1,
        row_labels: Sequence[str] = (),
        col_labels: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        """Provide a rectangular lattice of :class:`Cell` objects.

        The grid is responsible for:

        * creating the underlying ``np.ndarray`` of ``Cell`` instances,
        * arranging the rectangle placeholders in a Manim ``VGroup``,
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
        **kwargs
            Additional keyword arguments forwarded to the base ``Mobject``.

        Attributes
        ----------
        grid
            The ``VGroup`` containing the Rectangle objects defining each cell boundary.
        mobs
            A proxy giving access to the ``mob`` attribute of each cell. Supports
            read and write operations through ``__getitem__`` and ``__setitem__``.
        olds
            A proxy giving access to the ``old`` attribute of each cell. Supports
            read-only operation through ``__getitem__``.
        """
        super().__init__(**kwargs)

        num_rows, num_cols = len(row_heights), len(col_widths)
        self._row_heights = row_heights
        self._col_widths = col_widths
        self._buff = self._normalize_buff(buff)
        self._margin = self._normalize_margin(margin)

        _row_labels = self._prepare_labels(row_labels, num_rows)
        _col_labels = self._prepare_labels(col_labels, num_cols)
        self._label_mapper = LabelMapper(_row_labels, _col_labels)

        self._cells, self.grid = self._prepare_grid(
            num_rows, num_cols, row_heights, col_widths, self._buff
        )
        self.add(self.grid)

        self.mobs = MobsProxy(self, margin=self._margin)
        self.olds = OldsProxy(self)
        self.tags = TagsProxy(self)

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

    def _prepare_grid(
        self,
        num_rows: int,
        num_cols: int,
        row_heights: Sequence[float],
        col_widths: Sequence[float],
        buff: tuple[float, float],
    ) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.object_]], m.VGroup]:
        """Create the internal ``Cell`` matrix and the visual ``VGroup``.

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
                    stroke_opacity=0,
                    fill_opacity=0,
                )
                cells[i, j] = Cell(self, rect=rect)

        grid = m.VGroup(cell.rect for cell in cells.ravel())
        grid.arrange_in_grid(
            rows=num_rows,
            cols=num_cols,
            buff=buff,
            aligned_edge=m.UP,
        )
        return cells, grid

    @property
    def has_uniform_rows(self) -> bool:
        """Return ``True`` iff all the grid rows have the same height."""
        return len(set(self._row_heights)) == 1

    @property
    def has_uniform_cols(self) -> bool:
        """Return ``True`` iff all the grid cols have the same width."""
        return len(set(self._col_widths)) == 1

    def scroll(self, direction: Vector3DLike, step: int) -> Self:
        """Scroll the grid horizontally and/or vertically.

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
        """
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
        self.shift(offset)
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
            [self._col_widths[0], self._row_heights[0], 0.0]
        ) + np.array([*self._buff, 0.0])

        offset = (
            one_cell_offset
            * -1  # Scrolling UP means shifting DOWN.
            * direction
            * step
        )
        return np.array(offset)
