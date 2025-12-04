from typing import Literal, cast, overload

import numpy as np

from manim_grid.exceptions import GridLabelError
from manim_grid.typing import (
    Index,
    IntArrayIndex,
    IntArrayKey,
    Key,
    ListKey,
    MaskArrayIndex,
    MaskArrayKey,
    NpIndex,
    NpIntArrayIndex,
    NpIntArrayKey,
    NpKey,
    NpListKey,
    NpMaskArrayIndex,
    NpMaskArrayKey,
    NpSingleKey,
    NpSliceKey,
    NpStrArrayIndex,
    NpStrArrayKey,
    SingleKey,
    SliceKey,
    StrArrayIndex,
    StrArrayKey,
    is_1d_str_key,
    is_2d_str_index,
)


class LabelMapper:
    """Translate user-friendly labels into integer indices.

    The mapper knows how to turn **row** and **column** labels (provided as dictionaries
    ``{label: int}``) into the integer positions that numpy expects.

    Parameters
    ----------
    row_labels
        A mapping from row label to integer position.

    col_labels
        A mapping from column label to integer position.

    Note
    ----
        A ``key`` is a single allowed expression for row or column indexing
        (int, str, list[int|str], slice[int|str], 1D-array[bool|int|str].).
        An `index` is an allowed '2D' expression to be used to index a grid
        (key, tuple[key, key], 2D-array[bool|int|str].).
        A key is also an index as it will ultimately be resolved to ``(key, :)``.
    """

    def __init__(self, row_labels: dict[str, int], col_labels: dict[str, int]) -> None:
        self.row_labels = row_labels
        self.col_labels = col_labels

    @overload
    def map_index(self, index: Key) -> NpKey: ...

    @overload
    def map_index(self, index: tuple[Key, Key]) -> tuple[NpKey, NpKey]: ...

    @overload
    def map_index(self, index: MaskArrayIndex) -> NpMaskArrayIndex: ...

    @overload
    def map_index(self, index: IntArrayIndex) -> NpIntArrayIndex: ...

    @overload
    def map_index(self, index: StrArrayIndex) -> NpStrArrayIndex: ...

    def map_index(self, index: Index) -> NpIndex:
        """Resolve any supported index expression to a numpy-compatible index.

        Parameters
        ----------
        index
            An index expression. It may be a scalar key, a slice, a list of keys a 1D
            numpy array, a 2-tuple of the previous forms or a 2D numpy array.

        Returns
        -------
        NpIndex
            The resolved index. The concrete type mirrors the input, with strings mapped
            to integers.
        """
        if isinstance(index, tuple) and len(index) == 2:
            row_key, col_key = index
            return (
                self._map_key(row_key, self.row_labels, "row"),
                self._map_key(col_key, self.col_labels, "col"),
            )

        if isinstance(index, np.ndarray):
            if is_1d_str_key(index):
                return self._map_key(index, self.row_labels, "row")

            if is_2d_str_index(index):
                row_labels_arr = index[:, 0]
                col_labels_arr = index[:, 1]
                row_ints = self._map_key(row_labels_arr, self.row_labels, "row")
                col_ints = self._map_key(col_labels_arr, self.col_labels, "col")
                combined = np.column_stack((row_ints, col_ints))
                return combined

            # Bool or Int, 1D or 2D: return as is.
            return cast(
                MaskArrayKey | IntArrayKey | MaskArrayIndex | IntArrayIndex, index
            )

        # Everything else is a key that will ultimately be mapped to (row, :).
        return self._map_key(index, self.row_labels, "row")

    @overload
    def _map_key(
        self, key: SingleKey, label_map: dict[str, int], dim_name: Literal["row", "col"]
    ) -> NpSingleKey: ...

    @overload
    def _map_key(
        self, key: ListKey, label_map: dict[str, int], dim_name: Literal["row", "col"]
    ) -> NpListKey: ...

    @overload
    def _map_key(
        self, key: SliceKey, label_map: dict[str, int], dim_name: Literal["row", "col"]
    ) -> NpSliceKey: ...

    @overload
    def _map_key(
        self,
        key: MaskArrayKey,
        label_map: dict[str, int],
        dim_name: Literal["row", "col"],
    ) -> NpMaskArrayKey: ...

    @overload
    def _map_key(
        self,
        key: IntArrayKey,
        label_map: dict[str, int],
        dim_name: Literal["row", "col"],
    ) -> NpIntArrayKey: ...

    @overload
    def _map_key(
        self,
        key: StrArrayKey,
        label_map: dict[str, int],
        dim_name: Literal["row", "col"],
    ) -> NpStrArrayKey: ...

    def _map_key(
        self,
        key: Key,
        label_map: dict[str, int],
        dim_name: Literal["row", "col"],
    ) -> NpKey:
        """Resolve a key (an index component) to its integer representation.

        Parameters
        ----------
        key
            The key to resolve. It can be a scalar key (int or str), a slice, a list or
            a 1D numpy array.

        label_map
            Mapping from string label to integer position for the dimension indicated
            by ``dim_name``.

        dim_name
            The considered dimension (one of "row" or "col"); used for error messages.

        Returns
        -------
        NpKey
            The resolved key. The concrete type mirrors the input, with strings mapped
            to integers.

        Raises
        ------
        GridLabelError
            If a string label is not present in ``label_map``.
        """
        if isinstance(key, str):
            if key not in label_map:
                raise GridLabelError(f"{dim_name.title()} label {key!r} not defined.")
            return label_map[key]

        if isinstance(key, slice):
            # Recursively map start / stop.
            return slice(
                self._map_key(key.start, label_map, dim_name)
                if key.start is not None
                else None,
                self._map_key(key.stop, label_map, dim_name)
                if key.stop is not None
                else None,
                key.step,
            )

        if isinstance(key, np.ndarray):
            if is_1d_str_key(key):
                vec_lookup = np.vectorize(lambda s: label_map.get(s, -1))
                mapped = vec_lookup(key).astype(int)
                missing = key[mapped == -1]
                if missing.size:
                    raise GridLabelError(
                        f"{dim_name.title()} label{'s' if missing.size > 1 else ''} "
                        f"not defined: {', '.join(missing)}"
                    )
                return cast(IntArrayKey, mapped)
            else:
                # Bool or Int: return as is.
                return cast(MaskArrayKey | IntArrayKey, key)

        if isinstance(key, list):
            # List: map each element recursively.
            return [self._map_key(k, label_map, dim_name) for k in key]

        # Anything else (int, bool, etc.) is already numpy-compatible.
        return key
