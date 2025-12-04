import numpy as np
import pytest

from manim_grid.exceptions import GridLabelError
from manim_grid.labels import LabelMapper


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
@pytest.fixture
def simple_mapper():
    row_labels = {"A": 0, "B": 1, "C": 2}
    col_labels = {"X": 0, "Y": 1, "Z": 2}
    return LabelMapper(row_labels, col_labels)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_integer_keys_pass_through(simple_mapper):
    assert simple_mapper.map_index(2) == 2
    assert simple_mapper.map_index((1, 0)) == (1, 0)


def test_simple_index_is_resolved(simple_mapper):
    assert simple_mapper.map_index("B") == 1
    assert simple_mapper.map_index(("A", "Z")) == (0, 2)
    assert simple_mapper.map_index(("B", 2)) == (1, 2)
    assert simple_mapper.map_index((0, "Z")) == (0, 2)


def test_mixed_tuple(simple_mapper):
    mask = np.array([True, False, True])
    bool_str = simple_mapper.map_index((mask, "Z"))
    assert isinstance(bool_str[0], np.ndarray)
    assert bool_str[0].dtype == np.bool_
    assert bool_str[1] == 2
    np.testing.assert_array_equal(bool_str[0], mask)


def test_list_of_keys(simple_mapper):
    lst = ["A", "C"]
    assert simple_mapper.map_index(lst) == [0, 2]
    mixed = [-1, "B"]
    assert simple_mapper.map_index(mixed) == [-1, 1]


def test_slice(simple_mapper):
    slc = slice("A", "C")
    assert simple_mapper.map_index(slc) == slice(0, 2)


def test_1d_boolean_mask(simple_mapper):
    mask = np.array([True, False, True])
    out = simple_mapper.map_index(mask)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, mask)


def test_1d_int_array(simple_mapper):
    arr = np.arange(3, dtype=int)
    out = simple_mapper.map_index(arr)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int_
    np.testing.assert_array_equal(out, arr)


def test_1d_string_array_is_converted_to_ints(simple_mapper):
    str_arr = np.array(["A", "C", "B"])
    out = simple_mapper.map_index(str_arr)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int_
    np.testing.assert_array_equal(out, np.array([0, 2, 1]))


def test_1d_string_array_with_missing_labels_raises(simple_mapper):
    str_arr = np.array(["MISSING1", "C", "MISSING2"])
    with pytest.raises(
        GridLabelError, match="Row labels not defined: MISSING1, MISSING2"
    ):
        simple_mapper.map_index(str_arr)


def test_2d_boolean_mask(simple_mapper):
    mask = np.array([[True, False, True], [False, True, False]], dtype=bool)
    out = simple_mapper.map_index(mask)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, mask)


def test_2d_int_pair_array(simple_mapper):
    pairs = np.array([[0, 2], [0, 1]])
    out = simple_mapper.map_index(pairs)
    expected = np.array([[0, 2], [0, 1]], dtype=int)
    np.testing.assert_array_equal(out, expected)


def test_2d_str_pair_array(simple_mapper):
    pairs = np.array([["A", "Y"], ["C", "X"]])
    out = simple_mapper.map_index(pairs)
    expected = np.array([[0, 1], [2, 0]], dtype=int)
    np.testing.assert_array_equal(out, expected)


def test_invalid_string_label_raises(simple_mapper):
    with pytest.raises(GridLabelError):
        simple_mapper.map_index("UNKNOWN")

    with pytest.raises(GridLabelError):
        simple_mapper.map_index(("A", "UNKNOWN"))
