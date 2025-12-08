from abc import abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

import numpy as np

from manim_grid.typing import (
    MaskArrayIndex,
    ScalarIndex,
    SequenceIndex,
    is_scalar_index,
    is_sequence_index,
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

GScalarValT = TypeVar("GScalarValT")
"""Describe the scalar value type returned by ``__getitem__``."""

GSequenceValT = TypeVar("GSequenceValT")
"""Describe the sequence value type returned by ``__getitem__``."""

SScalarIdxT = TypeVar("SScalarIdxT")
"""Describe an additional scalar index form for ``__setitem``."""

SSequenceIdxT = TypeVar("SSequenceIdxT")
"""Describe an additional sequence index form for ``__setitem``."""

SScalarValT = TypeVar("SScalarValT")
"""Describe the scalar value type set through ``__setitem__``."""

SSequenceValT = TypeVar("SSequenceValT")
"""Describe the sequence value type set through ``__setitem__``."""

MISSING = object()
"""A sentinel value for missing attributes."""


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

    def mask(
        self, *, predicate: Callable[[Any], bool] | None = None, **kwargs: Any
    ) -> MaskArrayIndex:
        """Return a boolean ndarray with the same shape as the Cell matrix.

        This can be used as a boolean mask on any proxy to filter the selected objects.
        This method offers 2 ways to filter objects: a predicate and keyword arguments.
        Both conditions must be satisfied for an object to be included in the selection
        (i.e. for its index in the generated mask to be ``True``).

        Parameters
        ----------
        predicate
            A callable receiving the stored object and returning a boolean. The objects
            in each Cell will not be selected if the returned value is ``False``.
        kwargs
            Key/value pairs describing object attributes and values that must also be
            satisfied for the object to be selected. If an object does not have the
            ``key`` attribute or if its value does not correspond to the provided
            ``value``, it will not be selected.
        """
        values = np.vectorize(lambda cell: getattr(cell, self._attr), otypes=[object])(
            self._grid._cells
        )

        if predicate is None and not kwargs:
            raise ValueError("Provide a predicate or at least one keyword filter.")

        def combine(obj: object) -> bool:
            selected = True
            if predicate is not None:
                selected = selected and predicate(obj)
            for key, value in kwargs.items():
                selected = selected and getattr(obj, key, MISSING) == value
            return selected

        return cast(MaskArrayIndex, np.vectorize(combine, otypes=[bool])(values))


class ReadableProxy(
    Generic[GScalarIdxT, GSequenceIdxT, GScalarValT, GSequenceValT], _BaseProxy
):
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
    def __getitem__(self, index: ScalarIndex | GScalarIdxT) -> GScalarValT: ...

    @overload
    def __getitem__(self, index: SequenceIndex | GSequenceIdxT) -> GSequenceValT: ...

    def __getitem__(
        self, index: ScalarIndex | GScalarIdxT | SequenceIndex | GSequenceIdxT
    ) -> GScalarValT | GSequenceValT:
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
        GScalarValT | GSequenceValT
            GScalarValT if the indexed selection resolves to one Cell, otherwise
            GSequenceValT.
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

    @abstractmethod
    def _postprocess_get(
        self, subarray: "Cell | np.ndarray", **kwargs: Any
    ) -> GScalarValT | GSequenceValT:
        """Convert the raw ``subarray`` into the expected return type.

        Parameters
        ----------
        subarray
            Result of the numpy selector applied to ``self._grid._cells``.
            It may be a ``Cell`` or an ``ndarray`` of ``Cell`` objects.

        **kwargs
            Keyword arguments forwarded from ``_preprocess_get``.

        Returns
        -------
        GScalarValT | GSequenceValT
            Dependending on the indexed selection contained in ``subarray``.
        """
        ...


class WriteableProxy(
    Generic[SScalarIdxT, SSequenceIdxT, SScalarValT, SSequenceValT], _BaseProxy
):
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
    def __setitem__(
        self, index: ScalarIndex | SScalarIdxT, value: SScalarValT
    ) -> None: ...

    @overload
    def __setitem__(
        self, index: SequenceIndex | SSequenceIdxT, value: SSequenceValT
    ) -> None: ...

    def __setitem__(
        self,
        index: ScalarIndex | SScalarIdxT | SequenceIndex | SSequenceIdxT,
        value: SScalarValT | SSequenceValT,
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
            Value(s) to store. For a scalar cell a ``SScalarValT`` value is required;
            for multiple cells, a ``SSequenceValT`` value must be supplied.
        """
        idx, value, kwargs = self._preprocess_set(index, value)
        np_index = self._grid._label_mapper.map_index(idx)
        selector = np.index_exp[np_index]
        subarray = self._grid._cells[cast(Any, selector)]
        self._postprocess_set(subarray, value, **kwargs)

    def _preprocess_set(
        self,
        index: ScalarIndex | SScalarIdxT | SequenceIndex | SSequenceIdxT,
        value: SScalarValT | SSequenceValT,
    ) -> tuple[
        ScalarIndex | SequenceIndex, SScalarValT | SSequenceValT, dict[str, Any]
    ]:
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
        value: SScalarValT | SSequenceValT,
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
