from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, Never, TypeVar, cast, overload

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

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid

"""
These TypeVars describe the **additional** allowed indexes types for getting and
setting through proxies, as well as the value types for each proxy.
The indexing forms common to all proxies are already described in ``GScalarIndex`` and
``GSequenceIndex`` defined in :mod:`typing`.
For instance, :class:`MobsProxy` supports the 'Aligned' variants for indexing (also
defined in :mod:`typing.py`), and gets and sets :class:`manim.Mobject` instances
through :method:`_Proxy.__getitem__` and :method:`_Proxy.__setitem__`.
"""
GScalarIdxT = TypeVar("GScalarIdxT")
"""Describe an additional scalar index form for ``__getitem``."""

GSequenceIdxT = TypeVar("GSequenceIdxT")
"""Describe an additional sequence index form for ``__getitem``."""

GValT = TypeVar("GValT")
"""Describe the scalar value type returned by ``__getitem__``."""

SScalarIdxT = TypeVar("SScalarIdxT")
"""Describe an additional scalar index form for ``__setitem``."""

SSequenceIdxT = TypeVar("SSequenceIdxT")
"""Describe an additional sequence index form for ``__setitem``."""

SValT = TypeVar("SValT")
"""Describe the scalar value type set through ``__setitem__``."""


class _BaseProxy:
    """Base class for all proxy objects.

    A proxy is a thin façade that forwards attribute access to the underlying
    :class:`~manim_grid.grid.Cell` objects stored inside a
    :class:`~manim_grid.grid.Grid`. Concrete proxies specialize the behaviour for
    reading, writing, or both.

    Parameters
    ----------
    grid
        The parent :class:`~manim_grid.grid.Grid` instance that owns the ``_cells``
        matrix.
    attr
        Name of the attribute on :class:`~manim_grid.grid.Cell` that the proxy
        manipulates (e.g. ``"mob"``, ``"old"``, ``"tags"``).

    See Also
    --------
    _ReadableProxy : read-only proxy mixin.
    _WriteableProxy : write-only proxy mixin.
    """

    def __init__(self, grid: "Grid", attr: str) -> None:
        self._grid = grid
        self._attr = attr

    def __str__(self) -> str:
        vec = np.vectorize(
            lambda cell: str(getattr(cell, self._attr)), otypes=[np.str_]
        )
        return str(vec(self._grid._cells[:]))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} of size {self._grid._cells.shape}>"


class _ReadableProxy(Generic[GScalarIdxT, GSequenceIdxT, GValT], _BaseProxy):
    """Mixin that implements read-only indexing for a proxy.

    Sub-classes inherit ``__getitem__`` and the default implementations of
    ``_preprocess_get`` and ``_postprocess_get``. The generic type variables convey
    the relationship between any additional index type and the value type that the
    proxy returns.

    Parameters
    ----------
    grid
        Parent grid that owns the underlying ``_cells`` matrix.
    attr
        Name of the attribute on ``Cell`` that should be read.

    See Also
    --------
    _WriteableProxy : counterpart providing ``__setitem__``.
    """

    @overload
    def __getitem__(self, index: ScalarIndex | GScalarIdxT) -> GValT: ...

    @overload
    def __getitem__(self, index: SequenceIndex | GSequenceIdxT) -> list[GValT]: ...

    def __getitem__(
        self, index: ScalarIndex | GScalarIdxT | SequenceIndex | GSequenceIdxT
    ) -> GValT | list[GValT]:
        """Retrieve the attribute value(s) for *index*.

        This method performs three steps:

        1. Normalize the user supplied *index* via ``_preprocess_get``.
        2. Translate the normalized index into a numpy selector using the grid’s
           ``LabelMapper``.
        3. Extract the underlying ``Cell`` objects and delegate to ``_postprocess_get``
           for the final conversion.

        Parameters
        ----------
        index
            It may be an index specification understood by :class:`LabelMapper` or a
            custom index type described by ``GScalarIdxT`` and ``GSequenceIdxT``.

        Returns
        -------
        GValT | list[GValT]
            Single value when the selector resolves to one cell, otherwise a list of
            values ordered row-major.
        """
        idx, kwargs = self._preprocess_get(index)
        np_index = self._grid._label_mapper.map_index(idx)
        selector = np.index_exp[np_index]
        subarray = self._grid._cells[cast(Any, selector)]
        return self._postprocess_get(subarray, **kwargs)

    def _preprocess_get(
        self,
        index: ScalarIndex | GScalarIdxT | SequenceIndex | GSequenceIdxT,
    ) -> tuple[ScalarIndex | SequenceIndex, dict[str, Any]]:
        """Validate and transform *index* before it reaches the label mapper.

        The default implementation simply returns ``(index, {})`` after asserting that
        the index conforms to the type expected by the LabelMapper (i.e. any custom
        index format must be removed).

        Parameters
        ----------
        index
            Raw user supplied index.

        Returns
        -------
        tuple
            ``(clean_index, extra_kwargs)`` where ``extra_kwargs`` is forwarded to
            ``_postprocess_get`` for additional context.

        Raises
        ------
        AssertionError
            If ``index`` is neither a scalar nor a sequence index according to
            :func:`manim_grid.typing.is_scalar_index` /
            :func:`manim_grid.typing.is_sequence_index`.
        """
        assert is_scalar_index(index) or is_sequence_index(index)
        return index, {}

    def _postprocess_get(
        self, subarray: "Cell | np.ndarray", **_: Any
    ) -> GValT | list[GValT]:
        """Convert the raw ``subarray`` into the expected return type.

        Parameters
        ----------
        subarray
            Result of the numpy selector applied to ``self._grid._cells``.
            It may be a ``Cell`` or an ``ndarray`` of ``Cell`` objects.

        **_
            Placeholder for future keyword arguments - ignored by the default
            implementation.

        Returns
        -------
        GValT | list[GValT]
            Single value or list of values.
        """
        from manim_grid.grid import Cell

        if isinstance(subarray, Cell):
            return cast(GValT, getattr(subarray, self._attr))

        return [getattr(cell, self._attr) for cell in subarray.flat]


class _WriteableProxy(Generic[SScalarIdxT, SSequenceIdxT, SValT], _BaseProxy):
    """Mixin that implements write-only indexing for a proxy.

    Sub-classes must implement the abstract method ``_postprocess_set`` which receives
    the selected ``Cell`` objects (or an ``ndarray`` of them) and the user supplied
    value(s), and they may override ``_preprocess_set`` to customize the handling of
    custom index forms.
    The generic type variables describe the relationship between the additional index
    type and the value type that can be written.

    Parameters
    ----------
    grid
        Parent grid that owns the underlying ``_cells`` matrix.
    attr
        Name of the attribute on ``Cell`` that should be written.

    See Also
    --------
    _ReadableProxy : read-only counterpart.
    """

    @overload
    def __setitem__(self, index: ScalarIndex | SScalarIdxT, value: SValT) -> None: ...

    @overload
    def __setitem__(
        self, index: SequenceIndex | SSequenceIdxT, value: Sequence[SValT]
    ) -> None: ...

    def __setitem__(
        self,
        index: ScalarIndex | SScalarIdxT | SequenceIndex | SSequenceIdxT,
        value: SValT | Sequence[SValT],
    ) -> None:
        """Assign *value* to the cell(s) addressed by *index*.

        This method mirrors the workflow of ``__getitem__`` in :class:`_ReadableProxy`.

        1. ``_preprocess_set`` returns a cleaned index, a possibly transformed ``value``
           and a dictionary of extra keyword arguments.
        2. The cleaned index is turned into a numpy selector.
        3. ``_postprocess_set`` performs the actual mutation.

        Parameters
        ----------
        index
            It may be an index specification understood by :class:`LabelMapper` or a
            custom index type described by ``SScalarIdxT`` and ``SSequenceIdxT``.
        value
            Value(s) to store. For a scalar cell a single value is required;
            for multiple cells, a sequence whose length matches the number of selected
            cells must be supplied.
        """
        idx, value, kwargs = self._preprocess_set(index, value)
        np_index = self._grid._label_mapper.map_index(idx)
        selector = np.index_exp[np_index]
        subarray = self._grid._cells[cast(Any, selector)]
        self._postprocess_set(subarray, value, **kwargs)

    def _preprocess_set(
        self,
        index: ScalarIndex | SScalarIdxT | SequenceIndex | SSequenceIdxT,
        value: SValT | Sequence[SValT],
    ) -> tuple[ScalarIndex | SequenceIndex, SValT | Sequence[SValT], dict[str, Any]]:
        """Normalize *index* and *value* before they reach the grid.

        The default implementation simply validates that *index* is a scalar or sequence
        index and returns ``(index, value, {})``. Concrete proxies can extend this
        method to extract additional information (e.g. an alignment vector in
        MobsProxy).

        Parameters
        ----------
        index
            Raw user supplied index.
        value
            Raw value(s) supplied by the caller.

        Returns
        -------
        tuple
            ``(clean_index, transformed_value, extra_kwargs)`` where ``extra_kwargs``
            is a dictionary that will be forwarded to ``_postprocess_set``.

        Raises
        ------
        AssertionError
            If ``index`` is not a recognised scalar or sequence index.
        """
        assert is_scalar_index(index) or is_sequence_index(index)
        return index, value, {}

    @abstractmethod
    def _postprocess_set(
        self,
        subarray: "Cell | np.ndarray",
        value: SValT | Sequence[SValT],
        **kwargs: Any,
    ) -> None:
        """Perform the actual mutation of the selected cell(s).

        Parameters
        ----------
        subarray
            Target cell(s) to be mutated.
        value
            Value(s) to store in the cell(s).
        **kwargs
            Additional context supplied by ``_preprocess_set``.
        """
        ...


class MobsProxy(
    _ReadableProxy[Never, Never, m.Mobject],
    _WriteableProxy[AlignedScalarIndex, AlignedSequenceIndex, m.Mobject],
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


class OldsProxy(_ReadableProxy[Never, Never, m.Mobject]):
    """Read-only proxy that exposes the ``old`` attribute of each cell.

    The ``old`` attribute stores the *previous* :class:`manim.Mobject` that was present
    in the cell before the most recent insertion. It is useful for animations that need
    for instance to fade out or transform the former content.

    Parameters
    ----------
    grid
        Owning grid instance.
    attr
        Must be ``"old"`` - the name of the attribute this proxy reads.
    """

    ...
