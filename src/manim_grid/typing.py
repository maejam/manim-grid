from collections.abc import Sequence
from typing import Any, TypeAlias, TypeGuard

import numpy as np
from manim.typing import Vector3DLike

"""Indexing/Assignment.

A `Key` is a single allowed expression for row or column indexing
(int, str, list, 1D-array...).
An `Index` is an allowed '2D' expression to be used to index a grid
(tuple[key, key], 2D-array...).

Non-`Np` variants refer to valid user inputs.
`Np` variants refer to the resolved expressions as they will be fed to the underlying
numpy array.
"""

# Base types.
SingleKey: TypeAlias = int | str
"""A single row or column key accepted as user input."""

ListKey: TypeAlias = list[SingleKey]
"""A list of row or column keys accepted as user input."""

SliceKey: TypeAlias = slice
"""A row or column slice accepted as user input (int|str|None)."""

# NOTE: these type aliases for ndarrays enforce the number of dimensions but are not
# fully supported by type checkers yet.
# See: https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
MaskArrayKey: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
"""A 1D Array to be used as a boolean mask."""

IntArrayKey: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int_]]
"""A 1D Array of rows or colums keys accepted as user input."""

StrArrayKey: TypeAlias = np.ndarray[tuple[int], np.dtype[np.str_]]
"""A 1D Array of rows or colums labels accepted as user input."""

MaskArrayIndex: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool]]
"""A 2D Array to be used as a boolean mask."""

IntArrayIndex: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.int_]]
"""A 2D int array with (row, col) pairs accepted as user input."""

StrArrayIndex: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.str_]]
"""A 2D str array with (row, col) pairs accepted as user input."""

Key: TypeAlias = (
    SingleKey | ListKey | SliceKey | MaskArrayKey | IntArrayKey | StrArrayKey
)
"""Any valid type for a key accepted as user input."""

Index: TypeAlias = (
    Key | tuple[Key, Key] | MaskArrayIndex | IntArrayIndex | StrArrayIndex
)
"""Any valid type for an index accepted as user input."""

# Base types resolved to valid numpy indexes.
NpSingleKey: TypeAlias = int
"""A single row or column key resolved to a valid numpy index."""

NpListKey: TypeAlias = list[NpSingleKey]
"""A sequence of row or column keys resolved to a valid numpy index."""

NpSliceKey: TypeAlias = slice
"""A row or column slice resolved to a valid numpy index (int|None)."""

NpMaskArrayKey: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
"""A 1D Array to be used as a boolean mask resolved to a valid numpy index."""

NpIntArrayKey: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int_]]
"""A 1D Array of rows or colums keys resolved to a valid numpy index."""

NpStrArrayKey: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int_]]
"""A 1D Array of rows or colums labels resolved to a valid numpy index."""

NpMaskArrayIndex: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool]]
"""A 2D Array to be used as a boolean mask resolved to a valid numpy index."""

NpIntArrayIndex: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.int_]]
"""A 2D int array with (row, col) pairs resolved to a valid numpy index."""

NpStrArrayIndex: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.int_]]
"""A 2D str array with (row, col) pairs resolved to a valid numpy index."""

NpKey: TypeAlias = (
    NpSingleKey
    | NpListKey
    | NpSliceKey
    | NpMaskArrayKey
    | NpIntArrayKey
    | NpStrArrayKey
)
"""Any valid type for a key resolved to a valid numpy index."""

NpIndex: TypeAlias = (
    NpKey | tuple[NpKey, NpKey] | NpMaskArrayIndex | NpIntArrayIndex | NpStrArrayIndex
)
"""Any valid type for an index resolved to a valid numpy index."""

# Types related to different outputs.
ScalarIndex: TypeAlias = tuple[SingleKey, SingleKey]
"""An index that is resolved to a scalar object."""

BulkKey: TypeAlias = ListKey | SliceKey | MaskArrayKey | IntArrayKey | StrArrayKey
"""A key that is resolved to multiple objects."""

BulkIndex: TypeAlias = (
    SingleKey
    | BulkKey
    | tuple[SingleKey, BulkKey]
    | tuple[BulkKey, SingleKey]
    | tuple[BulkKey, BulkKey]
    | MaskArrayIndex
    | IntArrayIndex
    | StrArrayIndex
)
"""An index that is resolved to multiple objects."""

# Specialized indexes.
AlignedScalarIndex: TypeAlias = tuple[SingleKey, SingleKey, Vector3DLike]
"""A full index with alignment resolved to a scalar object."""

AlignedBulkIndex: TypeAlias = (
    tuple[SingleKey, Vector3DLike]
    | tuple[BulkKey, Vector3DLike]
    | tuple[SingleKey, BulkKey, Vector3DLike]
    | tuple[BulkKey, SingleKey, Vector3DLike]
    | tuple[BulkKey, BulkKey, Vector3DLike]
    | tuple[MaskArrayIndex, Vector3DLike]
    | tuple[IntArrayIndex, Vector3DLike]
    | tuple[StrArrayIndex, Vector3DLike]
)
"""A full index with alignment resolved to multiple objects."""


# TypeGuards.
def is_single_key(index: Any) -> TypeGuard[SingleKey]:
    """Return ``True`` iff ``index`` is compatible with ``SingleKey``."""
    return isinstance(index, (int, str))


def is_scalar_index(index: Any) -> TypeGuard[ScalarIndex]:
    """Return ``True`` iff ``index`` is compatible with ``ScalarIndex``."""
    return (
        isinstance(index, tuple)
        and len(index) == 2
        and all(is_single_key(k) for k in index)
    )


def _is_valid_dtype(arr: np.ndarray) -> bool:
    """Return ``True`` of the provided ``arr`` is of dtype bool|int|str."""
    return (
        np.issubdtype(arr.dtype, np.bool_)
        or np.issubdtype(arr.dtype, np.int_)
        or np.issubdtype(arr.dtype, np.str_)
    )


def is_bulk_key(index: Any) -> TypeGuard[BulkKey]:
    """Return ``True`` iff ``index`` is compatible with ``BulkKey``."""
    listk = isinstance(index, list) and all(is_single_key(k) for k in index)
    slicek = (
        isinstance(index, slice)
        and isinstance(index.start, (int | str | None))
        and isinstance(index.stop, (int | str | None))
        and isinstance(index.step, (int | None))
    )
    arrayk = (
        isinstance(index, np.ndarray) and index.ndim == 1 and _is_valid_dtype(index)
    )
    return listk or slicek or arrayk


def is_bulk_index(index: Any) -> TypeGuard[BulkIndex]:
    """Return ``True`` iff ``index`` is compatible with ``BulkIndex``."""
    tup_idx = (
        isinstance(index, tuple)
        and len(index) == 2
        and all((is_single_key(k) or is_bulk_key(k)) for k in index)
        and not all(is_single_key(k) for k in index)
    )
    array_idx = (
        isinstance(index, np.ndarray) and index.ndim == 2 and _is_valid_dtype(index)
    )
    return is_single_key(index) or is_bulk_key(index) or tup_idx or array_idx


def is_1d_str_key(key: Any) -> TypeGuard[StrArrayKey]:
    """Return ``True`` iff ``key`` is a 1D numpy array of strings."""
    return (
        isinstance(key, np.ndarray) and key.ndim == 1 and key.dtype.kind in {"U", "S"}
    )


def is_2d_str_index(index: Any) -> TypeGuard[StrArrayIndex]:
    """Return ``True`` iff ``index`` is a 2D numpy array of strings."""
    return (
        isinstance(index, np.ndarray)
        and index.ndim == 2
        and index.dtype.kind in {"U", "S"}
    )


def is_vector_3d_like(vec: Any) -> TypeGuard[Vector3DLike]:
    """Return ``True`` if ``vec`` is a 3-dimensional vector: ``[float, float, float]``.

    This represents anything which can be converted to a Vector3D numpy array.
    """
    seq = (
        isinstance(vec, Sequence)
        and len(vec) == 3
        and all(isinstance(v, (int, float)) for v in vec)
    )
    arr = (
        isinstance(vec, np.ndarray)
        and vec.ndim == 1
        and vec.shape[0] == 3
        and vec.dtype == np.float64
    )
    return seq or arr
