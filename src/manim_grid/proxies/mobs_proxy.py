from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import manim as m
import numpy as np
from blinker import signal
from manim.typing import Vector3D

from manim_grid.exceptions import GridValueError
from manim_grid.typing import (
    AlignedBulkIndex,
    AlignedScalarIndex,
    BulkIndex,
    ScalarIndex,
    is_bulk_index,
    is_scalar_index,
    is_vector_3d_like,
)

from .base import ReadableProxy, WriteableProxy

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid


class MobsProxy(
    ReadableProxy[m.Mobject, m.VGroup],
    WriteableProxy[m.Mobject, Sequence[m.Mobject] | m.Group | m.VGroup],
):
    """Proxy that provides read-write access to the ``mob`` attribute of each cell.

    This proxy supports the following calling conventions:

    1. ``grid.mobs[index]`` for scalar or bulk indexing.
    2. ``grid.mobs[row, col, Vector3D] = mob`` for a scalar assignment.
        The alignment vector can be omitted and will default to the last vector set
        for that cell (ORIGIN by default).
    3. ``grid.mobs[index, Vector3D] = [mob1, mob2, ...]`` for a bulk assignment.
       The number of values provided must equal the number of Cells selected by *index*.
       The alignment vector can be omitted and will default to the last vector set for
       each cell. If provided, the same alignment is applied to all assigned mobjects.

    Parameters
    ----------
    grid
        Parent grid that owns the underlying ``cells`` matrix.
    margin
        Margin vector used by :meth:`Cell.insert_mob` to offset the inserted mobject.

    See Also
    --------
    OldsProxy : read-only proxy exposing the previous ``mob`` value.

    """

    _attr = "mob"
    _bulk_container = m.VGroup

    def __init__(
        self,
        grid: "Grid",
        margin: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ) -> None:
        super().__init__(grid)
        self._margin = margin

    @overload
    def __setitem__(
        self, index: ScalarIndex | AlignedScalarIndex, value: m.Mobject
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        index: BulkIndex | AlignedBulkIndex,
        value: Sequence[m.Mobject] | m.Group | m.VGroup,
    ) -> None: ...

    def __setitem__(
        self,
        index: ScalarIndex | AlignedScalarIndex | BulkIndex | AlignedBulkIndex,
        value: m.Mobject | Sequence[m.Mobject] | m.Group | m.VGroup,
    ) -> None:
        idx, value, kwargs = self._preprocess_set(index, value)
        np_index = self._grid._label_mapper.map_index(idx)
        selector = np.index_exp[np_index]
        subarray = self._grid.cells[cast(Any, selector)]
        self._postprocess_set(subarray, value, **kwargs)
        mobs = [value] if isinstance(value, m.Mobject) else value
        signal("mobs_assigned").send(
            self._grid, grid=self._grid, index=index, mobs=mobs
        )

    def _preprocess_set(
        self,
        index: ScalarIndex | AlignedScalarIndex | BulkIndex | AlignedBulkIndex,
        value: m.Mobject | Sequence[m.Mobject] | m.Group | m.VGroup,
    ) -> tuple[
        ScalarIndex | BulkIndex, m.Mobject | Sequence[m.Mobject], dict[str, Any]
    ]:
        """Separate the optional alignment vector from the index.

        If *index* is a tuple whose last element satisfies
        :func:`manim_grid.typing.is_vector_3d_like`, that element is interpreted as the
        alignment vector and removed from the index that is passed to the label mapper.

        Parameters
        ----------
        index
            Raw user supplied index. It may or may not include an alignmnet vector.
            The alignment vector can be in the form of a 1D numpy array such as manim's
            direction constants (``UP``, ``DOWN``...), or a 3-tuple of numbers.
        value
            Raw value(s) supplied by the caller.

        Returns
        -------
        tuple
            ``(clean_index, value, {"align": alignment_vector})``.

        """
        if isinstance(index, tuple) and is_vector_3d_like(index[-1]):
            align = np.array(index[-1], dtype=np.float64)
            # Unpack index if it resolves to a 1-tuple after removing align.
            # Necessary to pass assertion below.
            idx = index[:-1][0] if len(index[:-1]) == 1 else index[:-1]
        else:
            align = None
            idx = cast(ScalarIndex | BulkIndex, index)
        assert is_scalar_index(idx) or is_bulk_index(idx), (
            f"The provided index is not valid: {index}."
        )
        return idx, value, {"align": align}

    def _postprocess_set(
        self,
        subarray: "Cell | np.ndarray",
        value: m.Mobject | Sequence[m.Mobject] | m.Group | m.VGroup,
        align: Vector3D | None = None,
        **_: Any,
    ) -> None:
        """Insert the supplied mobject(s) into the target cell(s).

        Parameters
        ----------
        subarray
            The cell or array of cells to be updated.
        value
            New mobject(s) to store.
        align
            Alignment vector passed to :meth:`Cell.insert_mob`.
        **_
            Placeholder for additional keyword arguments that may be supplied by
            ``_preprocess_set`` (currently only ``align``).

        Raises
        ------
        GridValueError
            If ``value`` is not a ``Mobject`` in the scalar case, or if the length of
            ``value`` does not match the number of selected cells in the bulk case.

        """
        from manim_grid.grid import Cell

        if isinstance(subarray, Cell):
            if not isinstance(value, m.Mobject):
                raise GridValueError(
                    "Only a single Mobject can be assigned to a single Cell."
                )
            subarray.insert_mob(value, align, self._margin)
            return

        if not isinstance(value, (Sequence, m.Group, m.VGroup)):
            raise GridValueError(
                "Bulk assignment requires a sequence or a (V)Group of Mobjects."
            )
        num_cells = int(np.prod(subarray.shape))
        num_vals = len(value)
        if num_cells != num_vals:
            raise GridValueError(
                f"Length mismatch between the selected cells ({num_cells}) "
                f"and the provided values ({num_vals})."
            )

        margin = self._margin
        for cell, mob in zip(subarray.flat, value, strict=True):
            cell.insert_mob(mob, align, margin)
