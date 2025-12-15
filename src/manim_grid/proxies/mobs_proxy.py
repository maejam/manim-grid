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


class MobsProxy(ReadableProxy[m.Mobject], WriteableProxy[m.Mobject]):
    """Proxy that provides read-write access to the ``mob`` attribute of each cell.

    This proxy supports the following calling conventions:

    1. ``grid.mobs[index]`` for scalar or bulk indexing.
    2. ``grid.mobs[row, col, Vector3D] = mob`` for a scalar assignment. The alignment
       vector can be omitted and will default to ``manim.ORIGIN``.
    3. ``grid.mobs[index, Vector3D] = [mob1, mob2, ...]`` for a bulk assignment.
       The number of values provided must equal the number of Cells selected by *index*.
       The alignment vector can be omitted and will default to ``manim.ORIGIN``. The
       same alignment is applied to all assigned mobjects.

    Parameters
    ----------
    grid
        Parent grid that owns the underlying ``_cells`` matrix.
    margin
        Margin vector used by :meth:`Cell.insert_mob` to offset the inserted mobject.

    See Also
    --------
    OldsProxy : read-only proxy exposing the previous ``mob`` value.
    """

    _attr: str = "mob"

    def __init__(
        self,
        grid: "Grid",
        margin: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ) -> None:
        super().__init__(grid)
        self._margin = margin

    @overload
    def __getitem__(self, index: ScalarIndex) -> m.Mobject: ...

    @overload
    def __getitem__(self, index: BulkIndex) -> m.VGroup: ...

    def __getitem__(
        self, index: ScalarIndex | BulkIndex
    ) -> m.Mobject | list[m.Mobject]:
        return cast(m.Mobject | m.VGroup, super().__getitem__(index))

    def _postprocess_get(
        self, subarray: "Cell | np.ndarray", **_: Any
    ) -> m.Mobject | m.VGroup:
        """Return a single Mobject in the scalar case or a VGroup of Mobjects."""
        from manim_grid.grid import Cell

        if isinstance(subarray, Cell):
            return cast(m.Mobject, getattr(subarray, self._attr))

        return m.VGroup(getattr(cell, self._attr) for cell in subarray.flat)

    @overload
    def __setitem__(
        self, index: ScalarIndex | AlignedScalarIndex, value: m.Mobject
    ) -> None: ...

    @overload
    def __setitem__(
        self, index: BulkIndex | AlignedBulkIndex, value: Sequence[m.Mobject]
    ) -> None: ...

    def __setitem__(
        self,
        index: ScalarIndex | AlignedScalarIndex | BulkIndex | AlignedBulkIndex,
        value: m.Mobject | Sequence[m.Mobject],
    ) -> None:
        super().__setitem__(index, value)

    def _preprocess_set(
        self,
        index: ScalarIndex | AlignedScalarIndex | BulkIndex | AlignedBulkIndex,
        value: m.Mobject | Sequence[m.Mobject],
    ) -> tuple[
        ScalarIndex | BulkIndex, m.Mobject | Sequence[m.Mobject], dict[str, Any]
    ]:
        """Separate the optional alignment vector from the index.

        If *index* is a tuple whose last element satisfies
        :func:`manim_grid.typing.is_vector_3d_like`, that element is interpreted as the
        alignment vector and removed from the index that is passed to the label mapper.
        When no vector is supplied, ``manim.ORIGIN`` is used.

        Parameters
        ----------
        index
            Raw user supplied index. It may or may not include an alignmnet vector.
            The alignment vector can be in the form of a 1D numpy array such as manim's
            direction constants (``UP``, ``DOWN``...), or a 3-tuple of numbers. Defaults
            to ``ORIGIN``.
        value
            Raw value(s) supplied by the caller.

        Returns
        -------
        tuple
            ``(clean_index, value, {"alignment": alignment_vector})``.
        """
        if isinstance(index, tuple) and is_vector_3d_like(index[-1]):
            alignment = np.array(index[-1], dtype=np.float64)
            idx = cast(ScalarIndex | BulkIndex, index[:-1])
        else:
            alignment = m.ORIGIN
            idx = cast(ScalarIndex | BulkIndex, index)
        assert is_scalar_index(idx) or is_bulk_index(idx)
        return idx, value, {"alignment": alignment}

    def _postprocess_set(
        self,
        subarray: "Cell | np.ndarray",
        value: m.Mobject | Sequence[m.Mobject],
        alignment: Vector3D = m.ORIGIN,
        **_: Any,
    ) -> None:
        """Insert the supplied mobject(s) into the target cell(s).

        Parameters
        ----------
        subarray
            The cell or array of cells to be updated.
        value
            New mobject(s) to store.
        alignment
            Alignment vector passed to :meth:`Cell.insert_mob`. Defaults to
            ``manim.ORIGIN`` (centered in the targeted cell(s)).
        **_
            Placeholder for additional keyword arguments that may be supplied by
            ``_preprocess_set`` (currently only ``alignment``).

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
            subarray.insert_mob(value, alignment, self._margin)
            return

        if not isinstance(value, Sequence):
            raise GridValueError("Bulk assignment requires a sequence of Mobjects.")
        num_cells = int(np.prod(subarray.shape))
        num_vals = len(value)
        if num_cells != num_vals:
            raise GridValueError(
                f"Length mismatch between the selected cells ({num_cells}) "
                f"and the provided values ({num_vals})."
            )

        margin = self._margin
        for cell, mob in zip(subarray.flat, value, strict=True):
            cell.insert_mob(mob, alignment, margin)
