import manim as m
import numpy as np
import pytest

from manim_grid.grid import Cell, Grid


# ----------------------------------------------------------------------
# Cell
# ----------------------------------------------------------------------
def test_cell_initial_state(dummy_mob):
    cell = Cell(dummy_mob)

    assert isinstance(cell.mob, m.Mobject)
    assert isinstance(cell.old, m.Mobject)
    assert cell.tags == {}


def test_cell_insert_mob_updates_old_and_mob(dummy_mob):
    cell = Cell(rect=dummy_mob)
    default = cell.mob

    first = dummy_mob.copy()
    second = dummy_mob.copy()

    cell.insert_mob(first, alignment=m.ORIGIN, margin=np.zeros(3))
    assert cell.mob is first
    assert cell.old is default

    cell.insert_mob(second, alignment=m.ORIGIN, margin=np.zeros(3))
    assert cell.mob is second
    assert cell.old is first


# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------
def test_prepare_grid_shapes(simple_grid):
    cells, vgroup = simple_grid._cells, simple_grid.grid
    assert cells.shape == (2, 3)
    assert all(isinstance(c, Cell) for c in cells.ravel())
    rects = [cell.rect for cell in cells.ravel()]
    assert list(vgroup) == rects


# ----------------------------------------------------------------------
# Grid - labels
# ----------------------------------------------------------------------
def test_prepare_labels_defaults():
    row_labels = Grid._prepare_labels((), 2)
    col_labels = Grid._prepare_labels((), 3)

    assert row_labels == {"1": 0, "2": 1}
    assert col_labels == {"1": 0, "2": 1, "3": 2}


def test_prepare_labels_custom():
    rows = ("top", "bottom")
    cols = ("left", "mid", "right")
    row_map = Grid._prepare_labels(rows, 2)
    col_map = Grid._prepare_labels(cols, 3)

    assert row_map == {"top": 0, "bottom": 1}
    assert col_map == {"left": 0, "mid": 1, "right": 2}


def test_label_mapper_is_populated(simple_grid):
    lm = simple_grid._label_mapper
    assert lm.row_labels == {"1": 0, "2": 1}
    assert lm.col_labels == {"1": 0, "2": 1, "3": 2}


def test_prepare_label_with_wrong_number_raises():
    with pytest.raises(
        ValueError, match="The number of labels should match the number of rows/columns"
    ):
        Grid._prepare_labels(["one", "two"], 3)


# ----------------------------------------------------------------------
# Grid - buffer
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "buff",
    [
        0,
        0.0,
        0.2,
        -0.2,
    ],
)
def test_normalize_buff_from_scalar(buff):
    result = Grid._normalize_buff(buff)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)
    expected = (float(buff), float(buff))
    assert result == expected


@pytest.mark.parametrize(
    "buff",
    [
        (0.0, 0.0),
        (1, 2),
        (3.5, 4.5),
        (True, False),
        (-1.2, 3.2, 0),
    ],
)
def test_normalize_buff_from_tuple(buff):
    result = Grid._normalize_buff(buff)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == float(buff[0])
    assert result[1] == float(buff[1])


@pytest.mark.parametrize(
    "buff",
    [
        "bad",
        (1, "two"),
        None,
    ],
)
def test_normalize_buff_invalid_input(buff):
    with pytest.raises(TypeError, match="Grid buffer should be a numeric value."):
        Grid._normalize_buff(buff)


# ----------------------------------------------------------------------
# Grid - margin
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "margin",
    [
        0,
        0.0,
        0.1,
        -0.1,
    ],
)
def test_normalize_margin_from_scalar(margin):
    result = Grid._normalize_margin(margin)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result.dtype == np.float64

    expected = np.array([margin, margin, 0.0], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "margin",
    [
        (0.0, 0.0),
        (1, 2),
        (3.5, 4.5),
        (3.5, -4.5),
    ],
)
def test_normalize_margin_from_tuple(margin):
    result = Grid._normalize_margin(margin)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result.dtype == np.float64

    expected = np.array([margin[0], margin[1], 0.0], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "margin",
    [
        "bad",
        (1, "two"),
        None,
    ],
)
def test_normalize_margin_invalid_input(margin):
    with pytest.raises(TypeError, match="Grid margin should be a numeric value."):
        Grid._normalize_margin(margin)


# # ----------------------------------------------------------------------
# # Optional – demonstrate the mask helper (the part you liked)
# # ----------------------------------------------------------------------
# def test_mask_helper_filters_by_attribute(simple_grid):
#     """The ``mask`` method must produce a boolean array that matches the query."""
#     red = DummyMobject()
#     blue = DummyMobject()
#     # Give the dummy objects a ``color`` attribute so the mask can inspect it
#     red.color = "RED"
#     blue.color = "BLUE"
#
#     # Populate the grid: first row red, second row blue
#     simple_grid.mobs[0, :] = [red, red, red]
#     simple_grid.mobs[1, :] = [blue, blue, blue]
#
#     # Mask for blue objects
#     blue_mask = simple_grid.mobs.mask(color="BLUE")
#     assert isinstance(blue_mask, np.ndarray)
#     assert blue_mask.shape == (2, 3)
#
#     # Expected pattern: first row False, second row True
#     expected = np.array([[False, False, False], [True, True, True]])
#     np.testing.assert_array_equal(blue_mask, expected)
#
#     # Using the mask to retrieve the objects should give us the three blues
#     blues = simple_grid.mobs[blue_mask]
#     assert blues == [blue, blue, blue]
#
#
# def test_mask_with_predicate(simple_grid):
#     """A callable predicate must also work."""
#     a = DummyMobject()
#     b = DummyMobject()
#     a.opacity = 0.2
#     b.opacity = 0.8
#
#     simple_grid.mobs[0, 0] = a
#     simple_grid.mobs[0, 1] = b
#
#     # Predicate selects objects with opacity > 0.5
#     mask = simple_grid.mobs.mask(predicate=lambda m: getattr(m, "opacity", 0) > 0.5)
#     assert mask.shape == (2, 3)  # whole grid shape
#     assert mask[0, 1] is True
#     assert mask[0, 0] is False
