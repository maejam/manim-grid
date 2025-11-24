import itertools
from collections.abc import Sequence
from typing import Any

import manim as m
import numpy as np
from manim.typing import Vector3D


class Grid(m.Mobject):
    """A grid canvas to ease mobjects positioning.

    Parameters
    ----------
    row_heights
        The height of each row, top-to-bottom.
    col_widths
        The width of each column left-to-right.
    buff
        The gap between grid cells.
        To specify a different buffer in the horizontal and vertical directions,
        a tuple of two values can be given - ``(row, col)``.
    margin
        Margin between a cell content and its border.
        To specify a different margin in the horizontal and vertical directions,
        a tuple of two values can be given - ``(row, col)``.
    kwargs
        Other keyword arguments passed to the :class:`manim.Mobject` constructor.

    Attributes
    ----------
    num_rows
        The number of rows, inferred from the length of ``row_heights``.
    num_cols
        The number of columns, inferred from the length of ``col_widths``.
    cells
        The grid cells as a list of rows, each row being a list of
        :class:`manim.Rectangle` elements for each column. Can be used to access
        each individual Rectangle by index (i.e. ``grid.cells[row_num][col_num]``).
    grid
        The same Rectangles in a flat :class:`manim.VGroup`. Can be used to access
        all the cells at once (e.g. to apply some style uniformly).
    _contents
        The content of each cell in a list of list similar to ``cells``. This should not
        be accessed directly. Prefer using the dedicated :method:`__getitem__` and
        :method:`__setitem__`.
    """

    def __init__(
        self,
        row_heights: Sequence[float],
        col_widths: Sequence[float],
        *,
        buff: float | tuple[float, float] = 0.0,
        margin: float | tuple[float, float] = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.row_heights = row_heights
        self.col_widths = col_widths
        self.num_rows = len(row_heights)
        self.num_cols = len(col_widths)
        self.buff = (buff, buff) if isinstance(buff, (int, float)) else buff
        self.margin = (
            np.array([margin, margin, 0.0])
            if isinstance(margin, (int, float))
            else np.array([float(margin[0]), float(margin[1]), 0.0])
        )
        self.kwargs = kwargs

        self.cells = [
            [
                m.Rectangle(
                    height=row_h,
                    width=col_w,
                    stroke_opacity=0,
                    fill_opacity=0,
                )
                for col_w in col_widths
            ]
            for row_h in row_heights
        ]
        self.grid = m.VGroup(itertools.chain.from_iterable(self.cells))
        self.grid.set(name="grid")

        self.grid.arrange_in_grid(
            rows=self.num_rows,
            cols=self.num_cols,
            buff=self.buff,
            aligned_edge=m.UP,
        )
        self.add(self.grid)

        self._contents: list[list[m.Mobject]] = [
            [m.Mobject() for _ in range(self.num_cols)] for _ in range(self.num_rows)
        ]

    def __getitem__(self, idx: tuple[int, int]) -> m.Mobject:
        """Retrieve the content of a cell.

        Parameters
        ----------
        idx
            A pair ``(row, col)`` identifying the cell whose content should be
            retrieved. Negative indices are interpreted in the usual Python fashion
            (e.g. ``-1`` refers to the last row/column).

        Returns
        -------
        The Mobject currently placed in the specified cell. If nothing has been
        assigned yet, a placeholder empty :class:`manim.Mobject` is returned.

        Raises
        ------
        IndexError
            If the supplied row or column index is outside the grid dimensions.

        Examples
        --------
        >>> grid = Grid([1, 1], [2, 2])
        >>> circle = Circle()
        >>> grid[0, 1] = circle # place a circle in the first row, second column
        >>> grid[0, 1] is circle
        True
        """
        i, j = idx
        if not (0 <= abs(i) < self.num_rows and 0 <= abs(j) < self.num_cols):
            raise IndexError(
                f"Cell ({i},{j}) out of range. "
                f"The grid shape is ({self.num_rows}x{self.num_cols})."
            )
        return self._contents[i][j]

    def __setitem__(
        self, idx: tuple[int, int] | tuple[int, int, Vector3D], mob: m.Mobject
    ) -> None:
        """Place a Mobject into a specific grid cell.

        The method accepts either a two-tuple ``(row, col)`` or a three-tuple
        ``(row, col, alignment)``. When the alignment vector is omitted, ``ORIGIN``
        is used, meaning the Mobject will be centered in the cell (both horizontally
        and vertically).

        Parameters
        ----------
        idx
            * ``(row, col)`` - coordinates of the target cell.
            * ``(row, col, alignment)`` - same as above, plus a 3‑D vector that
              determines which edge of the cell the Mobject should align to
              (e.g. ``UP``, ``DOWN``, ``UL``...).
            Negative indices are interpreted in the usual Python fashion
            (e.g. ``-1`` refers to the last row/column).
        mob
            The Mobject to insert. The Mobject is deliberatly only positionned into the
            cell and is not added to the scene to allow adding it through an Animation
            for instance. Similarly, the Mobject occupying this cell is not removed from
            the scene.

        Raises
        ------
        IndexError
            If the supplied row or column index is outside the grid dimensions.

        Examples
        --------
        >>> grid = Grid([1, 1], [2, 2])

        >>> # Centre a square in the second row, first column.
        >>> grid[1, 0] = Square()

        >>> # Add a triangle to the upper-right corner of the first cell.
        >>> grid[0, 0, UR] = m.Triangle()
        """
        if len(idx) == 2:
            i, j, alignment = *idx, m.ORIGIN
        else:
            i, j, alignment = idx

        if not (0 <= abs(i) < self.num_rows and 0 <= abs(j) < self.num_cols):
            raise IndexError(
                f"Cell ({i},{j}) out of range. "
                f"The grid shape is ({self.num_rows}x{self.num_cols})."
            )

        cell = self.cells[i][j]
        mob.move_to(cell, aligned_edge=alignment).shift(-alignment * self.margin)

        self._contents[i][j] = mob

    def __str__(self) -> str:
        return f"Grid ({self.num_rows}x{self.num_cols})"

    def __repr__(self) -> str:
        if self.kwargs:
            kwargs = ", " + ", ".join(
                f"{key}={value!r}" for key, value in self.kwargs.items()
            )
        else:
            kwargs = ""

        return (
            f"{type(self).__name__}(row_heights={self.row_heights!r}, "
            f"col_widths={self.col_widths!r}, buff={self.buff!r}, "
            f"margin={(float(self.margin[0]), float(self.margin[1]))!r}{kwargs})"
        )
