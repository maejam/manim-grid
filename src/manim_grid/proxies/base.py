from collections.abc import (
    Callable,
    Generator,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

import numpy as np

from manim_grid.exceptions import GridValueError
from manim_grid.helpers import MISSING
from manim_grid.typing import (
    BulkIndex,
    MaskArrayIndex,
    ScalarIndex,
    is_bulk_index,
    is_scalar_index,
)

if TYPE_CHECKING:
    from manim_grid.grid import Cell, Grid


S = TypeVar("S")
"""The scalar type the proxy handles (the type of ``_attr`` on ``Cell``)."""


class _BaseProxy(Generic[S]):
    """Base class for all proxy objects.

    A proxy is a façade that forwards attribute access to the underlying
    :class:`~manim_grid.grid.Cell` objects stored inside a
    :class:`~manim_grid.grid.Grid`. Concrete proxies specialize the behaviour for
    reading, writing, or both.

    Parameters
    ----------
    grid
        The parent :class:`~manim_grid.grid.Grid` instance that owns the ``cells``
        matrix.

    Attributes
    ----------
    _attr
        Name of the attribute on :class:`~manim_grid.grid.Cell` that the proxy
        manipulates (e.g. ``"mob"``, ``"old"``, ``"tags"``, ...).

    See Also
    --------
    _ReadableProxy : read-only proxy mixin.
    _WriteableProxy : write-only proxy mixin.

    """

    _attr: str

    def __init__(self, grid: "Grid") -> None:
        self._grid = grid
        self._scalar_type = type(getattr(self._grid.cells[0, 0], self._attr))

    def __str__(self) -> str:
        vec = np.vectorize(
            lambda cell: str(getattr(cell, self._attr)), otypes=[np.str_]
        )
        return str(vec(self._grid.cells[:]))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} of size {self._grid.cells.shape}>"

    def __len__(self) -> int:
        return len(self._grid.cells)

    def __iter__(self) -> Generator[S]:
        for cell in self._grid.cells.flat:
            yield getattr(cell, self._attr)

    def mask(
        self, *, predicate: Callable[[S], bool] | None = None, **kwargs: Any
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
            self._grid.cells
        )

        if predicate is None and not kwargs:
            raise ValueError(
                "You must provide a predicate or at least one keyword filter."
            )

        def combine(obj: S) -> bool:
            selected = True
            if predicate is not None:
                selected = selected and predicate(obj)
            for key, value in kwargs.items():
                selected = selected and getattr(obj, key, MISSING) == value
            return selected

        return cast(MaskArrayIndex, np.vectorize(combine, otypes=[bool])(values))


BO = TypeVar("BO")
"""The Bulk Output type the Readable proxy returns."""


class ReadableProxy(_BaseProxy[S], Generic[S, BO]):
    """Mixin that implements read-only indexing for a proxy.

    Attributes
    ----------
    _bulk_container
        The type of the container returned when indexing in bulk
        (e.g. ``list[S]``, ``VGroup``...). Used to instantiate the return value.

    See Also
    --------
    _WriteableProxy : counterpart providing ``__setitem__``.

    """

    _bulk_container: type

    @overload
    def __getitem__(self, index: ScalarIndex) -> S: ...

    @overload
    def __getitem__(self, index: BulkIndex) -> BO: ...

    def __getitem__(self, index: ScalarIndex | BulkIndex) -> S | BO:
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
            custom index type (e.g. including alignment).

        Returns
        -------
        S | BO
            The return type depends on each concrete proxy and on whether *index*
            resolves to a scalar value or a bulk selection.

        """
        idx, kwargs = self._preprocess_get(index)
        np_index = self._grid._label_mapper.map_index(idx)
        selector = np.index_exp[np_index]
        subarray = self._grid.cells[cast(Any, selector)]
        return self._postprocess_get(subarray, **kwargs)

    def _preprocess_get(
        self, index: ScalarIndex | BulkIndex
    ) -> tuple[ScalarIndex | BulkIndex, dict[str, Any]]:
        """Validate and transform *index* before it reaches the label mapper.

        The default implementation simply returns ``(index, {})`` after asserting that
        the index conforms to the type expected by the LabelMapper. It is the concrete
        proxy responsibility to make sure the index is cleaned-up if needed.

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
            If ``index`` is neither a scalar nor a bulk index according to
            :func:`manim_grid.typing.is_scalar_index` /
            :func:`manim_grid.typing.is_bulk_index`.

        """
        assert is_scalar_index(index) or is_bulk_index(index), (
            f"The provided index is not valid: {index}"
        )
        return index, {}

    def _postprocess_get(self, subarray: "Cell | np.ndarray", **kwargs: Any) -> S | BO:
        """Convert the raw ``subarray`` into the expected return type.

        Parameters
        ----------
        subarray
            Result of the numpy selector applied to ``self._grid.cells``.
            It may be a ``Cell`` or an ``ndarray`` of ``Cell`` objects.
        **kwargs
            Keyword arguments forwarded from ``_preprocess_get``.

        Returns
        -------
        S | B
            Depending on the concrete proxy and the indexed selection contained in
            ``subarray`` (scalar or bulk).

        """
        from manim_grid.grid import Cell

        if isinstance(subarray, Cell):
            return cast(S, getattr(subarray, self._attr))

        return cast(
            BO,
            self._bulk_container(
                *[getattr(cell, self._attr) for cell in subarray.flat]
            ),
        )


BI = TypeVar("BI")
"""The Bulk Input type(s) the Writeable proxy accepts."""


class WriteableProxy(_BaseProxy[S], Generic[S, BI]):
    """Mixin that implements write-only indexing for a proxy.

    See Also
    --------
    _ReadableProxy : read-only counterpart.

    """

    @overload
    def __setitem__(self, index: ScalarIndex, value: S) -> None: ...

    @overload
    def __setitem__(self, index: BulkIndex, value: BI) -> None: ...

    def __setitem__(self, index: ScalarIndex | BulkIndex, value: S | BI) -> None:
        """Assign *value* to the cell(s) addressed by *index*.

        This method mirrors the workflow of ``__getitem__`` in :class:`_ReadableProxy`.

        1. ``_preprocess_set`` returns a clean index, a possibly transformed ``value``
           and a dictionary of extra keyword arguments.
        2. The clean index is turned into a numpy selector.
        3. ``_postprocess_set`` performs the actual mutation.

        Parameters
        ----------
        index
            It may be an index specification understood by :class:`LabelMapper` or a
            custom index type.
        value
            Value(s) to store.
        """
        idx, value, kwargs = self._preprocess_set(index, value)
        np_index = self._grid._label_mapper.map_index(idx)
        selector = np.index_exp[np_index]
        subarray = self._grid.cells[cast(Any, selector)]
        self._postprocess_set(subarray, value, **kwargs)

    def _preprocess_set(
        self,
        index: ScalarIndex | BulkIndex,
        value: S | BI,
    ) -> tuple[ScalarIndex | BulkIndex, S | BI, dict[str, Any]]:
        """Normalize *index* and *value* before they reach the grid.

        The default implementation simply validates that *index* is a scalar or bulk
        index and returns ``(index, value, {})``. Concrete proxies can extend this
        method to extract additional information (e.g. an alignment vector in
        MobsProxy) or transform the passed value.

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
            If ``index`` is not a recognised scalar or bulk index.

        """
        assert is_scalar_index(index) or is_bulk_index(index), (
            "The provided index is not valid."
        )

        return index, value, {}

    def _postprocess_set(
        self,
        subarray: "Cell | np.ndarray",
        value: S | BI,
        **kwargs: Any,
    ) -> None:
        """Perform the actual mutation of the selected cell(s).

        This default implementation simply sets `_attr` to value even in the bulk case
        (the same value is assigned to all selected Cells). Override or extend for more
        complex logic.

        Parameters
        ----------
        subarray
            Target cell(s) to be mutated.
        value
            Value(s) to store in the cell(s).
        **kwargs
            Additional context supplied by ``_preprocess_set``.

        """
        from manim_grid.grid import Cell

        # scalar assignment
        if isinstance(subarray, Cell):
            if not isinstance(value, self._scalar_type):
                raise GridValueError(
                    f"Only a single {self._scalar_type.__name__} can be assigned to a "
                    "single Cell."
                )
            setattr(subarray, self._attr, value)
            return

        # bulk assignment with scalar value
        if isinstance(value, self._scalar_type):
            for cell in subarray.flat:
                setattr(cell, self._attr, value)
        else:
            raise GridValueError(
                f"Bulk assignment for {self._attr} expects a single value that "
                "will be assigned to all selected Cells."
            )
