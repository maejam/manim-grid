from typing import Any, TypeAlias, TypeGuard

import numpy as np
from manim.typing import Vector3D

"""Indexing.

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

SequenceKey: TypeAlias = ListKey | SliceKey | MaskArrayKey | IntArrayKey | StrArrayKey
"""A key that is resolved to a sequence of objects."""

SequenceIndex: TypeAlias = (
    SingleKey
    | SequenceKey
    | tuple[SingleKey, SequenceKey]
    | tuple[SequenceKey, SingleKey]
    | tuple[SequenceKey, SequenceKey]
    | MaskArrayIndex
    | IntArrayIndex
    | StrArrayIndex
)
"""An index that is resolved to a sequence of objects."""

# Specialized indexes.
AlignedScalarIndex: TypeAlias = tuple[SingleKey, SingleKey, Vector3D]
"""A full index with alignment resolved to a scalar object."""

AlignedSequenceIndex: TypeAlias = (
    tuple[SingleKey, Vector3D]
    | tuple[SequenceKey, Vector3D]
    | tuple[SingleKey, SequenceKey, Vector3D]
    | tuple[SequenceKey, SingleKey, Vector3D]
    | tuple[SequenceKey, SequenceKey, Vector3D]
    | tuple[MaskArrayIndex, Vector3D]
    | tuple[IntArrayIndex, Vector3D]
    | tuple[StrArrayIndex, Vector3D]
)
"""A full index with alignment resolved to a sequence of objects."""


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


def is_sequence_key(index: Any) -> TypeGuard[SequenceKey]:
    """Return ``True`` iff ``index`` is compatible with ``SequenceKey``."""
    listk = isinstance(index, list) and all(is_single_key(k) for k in index)
    slicek = (
        isinstance(index, slice)
        and isinstance(index.start, (int | str | None))
        and isinstance(index.stop, (int | str | None))
        and isinstance(index.step, (int | None))
    )
    arrayk = (
        isinstance(index, np.ndarray)
        and index.ndim == 1
        and index.dtype in {np.bool_, np.int_, np.str_}
    )
    return listk or slicek or arrayk


def is_sequence_index(index: Any) -> TypeGuard[SequenceIndex]:
    """Return ``True`` iff ``index`` is compatible with ``SequenceIndex``."""
    tup_idx = (
        isinstance(index, tuple)
        and len(index) == 2
        and all((is_single_key(k) or is_sequence_key(k)) for k in index)
        and not all(is_single_key(k) for k in index)
    )
    array_idx = (
        isinstance(index, np.ndarray)
        and index.ndim == 2
        and index.dtype in {np.bool_, np.int_, np.str_}
    )
    return is_single_key(index) or is_sequence_key(index) or tup_idx or array_idx


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


def is_vector_3d_compatible(vec: Any) -> TypeGuard[Vector3D]:
    """Return ``True`` if ``vec`` **could** be a ``Vector3D`` object.

    Any float64 numpy array with shape (3,) will pass this check.
    """
    return (
        isinstance(vec, np.ndarray)
        and vec.ndim == 1
        and vec.shape[0] == 3
        and vec.dtype == np.float64
    )
