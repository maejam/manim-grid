import manim as m
import numpy as np
import pytest

from manim_grid.exceptions import GridFrameError, GridShapeError, GridStencilError
from manim_grid.grid import Cell, EmptyMobject, Grid, Tags


# ----------------------------------------------------------------------
# Cell
# ----------------------------------------------------------------------
def test_cell_initial_state():
    cell = Cell(Grid([1], [1]), m.Rectangle(), 0, 1)

    assert isinstance(cell.mob, EmptyMobject)
    assert isinstance(cell.old, EmptyMobject)
    assert isinstance(cell.tags, Tags)
    assert isinstance(cell.rect, m.Rectangle)
    assert cell.row_index == 0
    assert cell.col_index == 1


def test_cell_insert_mob_updates_old_and_mob(dummy_mob):
    cell = Cell(Grid([1], [1]), m.Rectangle(), 0, 1)
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
    cells, vgroup = simple_grid.cells, simple_grid.lattice
    assert cells.shape == (2, 3)
    assert all(isinstance(c, Cell) for c in cells.ravel())
    rects = [cell.rect for cell in cells.ravel()]
    assert list(vgroup) == rects


# ----------------------------------------------------------------------
# Grid - alternative constructors
# ----------------------------------------------------------------------
@pytest.mark.parametrize(("num_rows", "num_cols"), [(1, 1), (1, 2), (10, 10)])
def test_fullscreen_grid_without_buffers(num_rows, num_cols):
    grid = Grid.fullscreen(num_rows, num_cols)
    assert sum(grid._row_heights) == pytest.approx(m.config.frame_height)
    assert sum(grid._col_widths) == pytest.approx(m.config.frame_width)


@pytest.mark.parametrize(
    ("num_rows", "num_cols", "buff"), [(1, 1, 0), (1, 2, 1), (10, 10, (1, 0.2))]
)
def test_fullscreen_grid_with_buffers(num_rows, num_cols, buff):
    grid = Grid.fullscreen(num_rows, num_cols, buff=buff)
    assert (
        sum(grid._row_heights) + grid._buff[1] * (num_rows - 1) == m.config.frame_height
    )
    assert (
        sum(grid._col_widths) + grid._buff[0] * (num_cols - 1) == m.config.frame_width
    )


@pytest.mark.parametrize(("num_rows", "num_cols"), [(0, 1), (1, -2)])
def test_fullscreen_grid_with_invalid_num_rows_cols_raises(num_rows, num_cols):
    with pytest.raises(GridShapeError, match="A Grid should have at least 1 row"):
        grid = Grid.fullscreen(num_rows, num_cols)


# ----------------------------------------------------------------------
# Grid - labels / numbers
# ----------------------------------------------------------------------
def test_prepare_labels_empty_labels():
    row_labels = Grid._prepare_labels((), 2)
    col_labels = Grid._prepare_labels((), 3)

    assert row_labels == {}
    assert col_labels == {}


def test_prepare_labels_custom():
    rows = ("top", "bottom")
    cols = ("left", "mid", "right")
    row_map = Grid._prepare_labels(rows, 2)
    col_map = Grid._prepare_labels(cols, 3)

    assert row_map == {"top": 0, "bottom": 1}
    assert col_map == {"left": 0, "mid": 1, "right": 2}


def test_label_mapper_is_populated(simple_grid):
    lm = simple_grid._label_mapper
    assert lm.row_labels == {}
    assert lm.col_labels == {}


def test_prepare_label_with_wrong_number_raises():
    with pytest.raises(
        ValueError, match="The number of labels should match the number of rows/columns"
    ):
        Grid._prepare_labels(["one", "two"], 3)


def test_row_col_numbers_are_correct(simple_grid: Grid):
    assert simple_grid._row_numbers == [0, 1]
    assert simple_grid._col_numbers == [0, 1, 2]


def test_labels_convenience_methods():
    g = Grid([1] * 4, [1], col_labels=["only"])
    assert g.row_labels(font_size=12) == []
    lab = g.col_labels(font_size=12)
    assert len(lab) == 1
    assert isinstance(lab[0], m.Text)
    assert lab[0].font_size == 12


def test_numbers_convenience_methods(simple_grid: Grid):
    rowno = simple_grid.row_numbers(font_size=14)
    colno = simple_grid.col_numbers(font_size=16)
    assert len(rowno) == 2
    assert len(colno) == 3
    assert all(isinstance(r, m.Text) for r in rowno)
    assert all(isinstance(c, m.Text) for c in colno)
    assert all(r.font_size == 14 for r in rowno)
    assert all(c.font_size == 16 for c in colno)


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
    with pytest.raises(TypeError, match="Grid buffer should be a numeric value"):
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
    with pytest.raises(TypeError, match="Grid margin should be a numeric value"):
        Grid._normalize_margin(margin)


# ----------------------------------------------------------------------
# Grid - scroll
# ----------------------------------------------------------------------
def test_scrolling_without_a_stencil_raises():
    g = Grid([1, 1], [1, 1, 1])
    with pytest.raises(GridStencilError, match="A grid without a stencil"):
        g.scroll(m.DOWN, 1)
    with pytest.raises(GridStencilError, match="A grid without a stencil"):
        g.free_scroll(m.DOWN, 1)


def test_vertical_scroll_non_uniform_rows_raises(simple_grid: Grid):
    simple_grid._row_heights = [1.5, 1.0]
    with pytest.raises(GridShapeError, match="the grid must have uniform"):
        simple_grid.scroll(m.DOWN, 2)


def test_horizontal_scroll_non_uniform_cols_raises(simple_grid: Grid):
    simple_grid._col_widths = [1.5, 1.0, 1.0]
    with pytest.raises(GridShapeError, match="the grid must have uniform"):
        simple_grid.scroll(m.LEFT, 2)


def test_horizontal_scroll_non_uniform_rows_does_not_raise(simple_grid: Grid):
    simple_grid._row_heights = [1.5, 1.0]
    simple_grid.scroll(m.LEFT, 2)


def test_vertical_scroll_non_uniform_cols_does_not_raise(simple_grid: Grid):
    simple_grid._col_widths = [1.5, 1.0, 1.0]
    simple_grid.scroll(m.UP, 2)


@pytest.mark.parametrize(
    ("direction", "step", "expected"),
    [
        (m.UP, 3, (0, -1 * (1 + 0.3) * 3, 0)),
        (m.DOWN, -3, (0, -1 * (1 + 0.3) * 3, 0)),
        (m.LEFT, 2, ((1.5 + 0.1) * 2, 0, 0)),
        (m.UL, 5, ((1.5 + 0.1) * 5, -1 * (1 + 0.3) * 5, 0)),
        ((2, 4, 1), 2, (-1 * (1.5 + 0.1) * 2 * 2, -1 * (1 + 0.3) * 2 * 4, 0 * 2 * 1)),
        (m.DOWN, 0, (0, 0, 0)),
    ],
)
def test_scroll_offset_is_correct(simple_grid: Grid, direction, step, expected):
    result = simple_grid._compute_scroll_offset(direction, step)
    np.testing.assert_array_equal(result, expected)


# ----------------------------------------------------------------------
# Grid - submobjects
# ----------------------------------------------------------------------
def test_new_grid_has_right_submobjects(simple_grid: Grid):
    assert simple_grid.rects[0, 0] in simple_grid.submobjects
    assert simple_grid.olds[0, 0] in simple_grid.submobjects
    assert simple_grid.mobs[0, 0] in simple_grid.submobjects
    assert simple_grid.submobjects[-2] is simple_grid.stencil
    assert simple_grid.submobjects[-1] is simple_grid.viewport


def test_grid_has_right_submobjects_after_assigning_mob(
    simple_grid: Grid,
):
    old = simple_grid.mobs[0, 0]
    c = m.Circle()
    simple_grid.mobs[0, 0] = c
    assert simple_grid.rects[0, 0] in simple_grid.submobjects
    assert simple_grid.olds[0, 0] is old
    assert old in simple_grid.submobjects
    assert simple_grid.mobs[0, 0] is c
    assert c not in simple_grid.submobjects
    assert simple_grid.submobjects[-2] is simple_grid.stencil
    assert simple_grid.submobjects[-1] is simple_grid.viewport


def test_grid_has_right_submobjects_after_assigning_group(
    simple_grid: Grid,
):
    r = m.Rectangle()
    c = m.Circle()
    t = m.Triangle()
    simple_grid.mobs[0] = [r, c, t]
    assert simple_grid.mobs[0, 0] is r
    assert simple_grid.mobs[0, 1] is c
    assert simple_grid.mobs[0, 2] is t
    assert r not in simple_grid.submobjects
    assert c not in simple_grid.submobjects
    assert t not in simple_grid.submobjects
    assert simple_grid.submobjects[-2] is simple_grid.stencil
    assert simple_grid.submobjects[-1] is simple_grid.viewport


def test_grid_has_right_submobjects_after_adding_and_removing_mob(simple_grid: Grid):
    r = m.Rectangle()
    c = m.Circle()
    t = m.Triangle()
    initial_len = len(simple_grid.submobjects)
    simple_grid.mobs[0] = [r, c, t]
    simple_grid.add(c)
    assert len(simple_grid.submobjects) == initial_len + 1
    assert c in simple_grid.submobjects
    assert r not in simple_grid.submobjects
    simple_grid.remove(c)
    assert len(simple_grid.submobjects) == initial_len
    assert c not in simple_grid.submobjects
    assert r not in simple_grid.submobjects


def test_grid_has_right_submobjects_after_adding_and_removing_group(simple_grid: Grid):
    r = m.Rectangle()
    c = m.Circle()
    t = m.Triangle()
    initial_len = len(simple_grid.submobjects)
    simple_grid.mobs[0] = [r, c, t]
    simple_grid.add(simple_grid.mobs[0, :-1])
    assert len(simple_grid.submobjects) == initial_len + 2
    assert r in simple_grid.submobjects
    assert c in simple_grid.submobjects
    assert t not in simple_grid.submobjects
    simple_grid.remove(simple_grid.mobs[0, :-1])
    assert len(simple_grid.submobjects) == initial_len
    assert c not in simple_grid.submobjects
    assert r not in simple_grid.submobjects


def test_grid_has_right_submobjects_after_adding_to_back_and_removing_mob(
    simple_grid: Grid,
):
    r = m.Rectangle()
    c = m.Circle()
    t = m.Triangle()
    initial_len = len(simple_grid.submobjects)
    simple_grid.mobs[0] = [r, c, t]
    simple_grid.add_to_back(c)
    assert len(simple_grid.submobjects) == initial_len + 1
    assert c is simple_grid.submobjects[0]
    assert r not in simple_grid.submobjects
    simple_grid.remove(c)
    assert len(simple_grid.submobjects) == initial_len
    assert c not in simple_grid.submobjects
    assert r not in simple_grid.submobjects


def test_grid_has_right_submobjects_after_adding_to_back_and_removing_group(
    simple_grid: Grid,
):
    r = m.Rectangle()
    c = m.Circle()
    t = m.Triangle()
    initial_len = len(simple_grid.submobjects)
    simple_grid.mobs[0] = [r, c, t]
    simple_grid.add_to_back(simple_grid.mobs[0, :-1])
    print(simple_grid.submobjects)
    assert len(simple_grid.submobjects) == initial_len + 2
    assert r is simple_grid.submobjects[0]
    assert c is simple_grid.submobjects[1]
    assert t not in simple_grid.submobjects
    simple_grid.remove(simple_grid.mobs[0, :-1])
    assert len(simple_grid.submobjects) == initial_len
    assert c not in simple_grid.submobjects
    assert r not in simple_grid.submobjects


# ----------------------------------------------------------------------
# Grid - frame
# ----------------------------------------------------------------------
def test_accesing_inexistent_frame_raises(simple_grid: Grid):
    with pytest.raises(GridFrameError, match="This Grid does not have a frame."):
        assert simple_grid.frame


def test_frame_without_stencil():
    g = Grid([1, 1], [1, 1, 1])
    assert g._frame is None
    r = m.SurroundingRectangle(g)
    g.frame = r
    gframe = g.frame
    assert isinstance(g.frame, m.Difference)
    assert np.array_equal(g.frame.get_center(), g.get_center())
    g.shift(m.RIGHT)
    assert np.array_equal(g.frame.get_center(), g.get_center())
    g.scale(0.1)
    assert np.array_equal(g.frame.get_center(), g.get_center())
    assert g.submobjects[-2] is gframe
    assert g.submobjects[-1] is g.viewport
    g.frame = None
    assert gframe not in g.submobjects
    assert g.submobjects[-1] is g.viewport


def test_frame_with_stencil(simple_grid: Grid):
    assert simple_grid._frame is None
    r = m.SurroundingRectangle(simple_grid.viewport)
    simple_grid.frame = r
    assert isinstance(simple_grid.frame, m.Difference)
    np.testing.assert_allclose(
        simple_grid.frame.get_center(), simple_grid.stencil.clip.get_center()
    )
    simple_grid.shift(m.RIGHT)
    np.testing.assert_allclose(
        simple_grid.frame.get_center(), simple_grid.stencil.clip.get_center()
    )
    simple_grid.scale(0.1)
    np.testing.assert_allclose(
        simple_grid.frame.get_center(), simple_grid.stencil.clip.get_center()
    )
    simple_grid.scroll(m.UP, 10000)
    np.testing.assert_allclose(
        simple_grid.frame.get_center(), simple_grid.stencil.clip.get_center()
    )
    assert not np.array_equal(simple_grid.frame.get_center(), simple_grid.get_center())

    assert simple_grid.submobjects[-3] is simple_grid.stencil
    assert simple_grid.submobjects[-2] is simple_grid.frame
    assert simple_grid.submobjects[-1] is simple_grid.viewport
    gframe = simple_grid.frame
    simple_grid.frame = None
    assert simple_grid.submobjects[-2] is simple_grid.stencil
    assert simple_grid.submobjects[-1] is simple_grid.viewport
    assert gframe not in simple_grid.submobjects
