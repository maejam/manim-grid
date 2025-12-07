from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Never,
    cast,
)

import manim as m
import numpy as np
from manim.typing import Vector3D

from manim_grid.exceptions import GridValueError
from manim_grid.typing import (
    AlignedScalarIndex,
    AlignedSequenceIndex,
    ScalarIndex,
    SequenceIndex,
    is_scalar_index,
    is_sequence_index,
    is_vector_3d_like,
)

from .base import ReadableProxy, WriteableProxy

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid


class MobsProxy(
    ReadableProxy[Never, Never, m.Mobject],
    WriteableProxy[AlignedScalarIndex, AlignedSequenceIndex, m.Mobject],
):
    """Proxy that provides read-write access to the ``mob`` attribute of each cell.

    The proxy supports two calling conventions:

    1. **Scalar assignment** – ``grid.mobs[row, col] = mob``
       The index may optionally be a tuple ``(row, col, alignment_vector)``;
       the alignment vector is passed to :meth:`Cell.insert_mob`.

    2. **Bulk assignment** – ``grid.mobs[row_slice, col_slice] = [mob1, mob2, …]``
       or ``grid.mobs[row_slice, col_slice, alignment_vector]``.
       The length of the supplied list must match the number of selected cells.

    Parameters
    ----------
    grid
        Parent grid that owns the underlying ``_cells`` matrix.
    attr
        Must be ``"mob"`` – the attribute name on ``Cell`` that stores the
        current :class:`manim.mobject.mobject.Mobject`.
    margin
        Margin vector used by :meth:`Cell.insert_mob` to offset the inserted mobject.

    See Also
    --------
    OldsProxy : read-only proxy exposing the previous ``mob`` value.
    """

    def __init__(
        self,
        grid: "Grid",
        attr: Literal["mob"],
        margin: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ) -> None:
        super().__init__(grid, attr)
        self._margin = margin

    def _preprocess_set(
        self,
        index: ScalarIndex | AlignedScalarIndex | SequenceIndex | AlignedSequenceIndex,
        value: m.Mobject | Sequence[m.Mobject],
    ) -> tuple[
        ScalarIndex | SequenceIndex, m.Mobject | Sequence[m.Mobject], dict[str, Any]
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
            idx = cast(ScalarIndex | SequenceIndex, index[:-1])
        else:
            alignment = m.ORIGIN
            idx = cast(ScalarIndex | SequenceIndex, index)
        assert is_scalar_index(idx) or is_sequence_index(idx)
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
