import manim as m
import numpy as np
import pytest

from manim_grid.grid import Grid


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
@pytest.fixture
def simple_grid() -> Grid:
    row_heights = [1.0, 2.0]
    col_widths = [0.5, 1.5, 2.5]
    return Grid(
        row_heights=row_heights,
        col_widths=col_widths,
        buff=0.0,
        margin=0.1,
        name="test-grid",
    )


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_grid_attributes(simple_grid: Grid) -> None:
    assert simple_grid.num_rows == 2
    assert simple_grid.num_cols == 3
    assert simple_grid.row_heights == [1.0, 2.0]
    assert simple_grid.col_widths == [0.5, 1.5, 2.5]
    assert simple_grid.buff == (0.0, 0.0)
    assert simple_grid.margin.all() == np.array([0.1, 0.1, 0.0]).all()


def test_cells_structure(simple_grid):
    assert isinstance(simple_grid.cells, list)
    assert len(simple_grid.cells) == simple_grid.num_rows
    for row in simple_grid.cells:
        assert isinstance(row, list)
        assert len(row) == simple_grid.num_cols
        for rect in row:
            assert isinstance(rect, m.Rectangle)

    assert len(simple_grid.grid) == simple_grid.num_rows * simple_grid.num_cols
    assert all(isinstance(cell, m.Rectangle) for cell in simple_grid.grid)


def test_cells_created_with_correct_sizes(simple_grid: Grid) -> None:
    for r, row_h in enumerate(simple_grid.row_heights):
        for c, col_w in enumerate(simple_grid.col_widths):
            cell = simple_grid.cells[r][c]
            assert cell.height == row_h
            assert cell.width == col_w


# ----------------------------------------------------------------------
# String representations
# ----------------------------------------------------------------------
def test_str_representation(simple_grid):
    assert str(simple_grid) == "Grid (2x3)"


def test_repr_without_kwargs(simple_grid):
    simple_grid.kwargs = {}
    rep = repr(simple_grid)
    assert rep == (
        "Grid(row_heights=[1.0, 2.0], col_widths=[0.5, 1.5, 2.5], "
        "buff=(0.0, 0.0), margin=(0.1, 0.1))"
    )


def test_repr_with_kwargs(simple_grid):
    rep = repr(simple_grid)
    assert rep == (
        "Grid(row_heights=[1.0, 2.0], col_widths=[0.5, 1.5, 2.5], "
        "buff=(0.0, 0.0), margin=(0.1, 0.1), name='test-grid')"
    )


# ----------------------------------------------------------------------
# Indexing / Assignment
# ----------------------------------------------------------------------
def test_getitem_returns_placeholder(simple_grid: Grid) -> None:
    placeholder = simple_grid[0, 0]
    assert isinstance(placeholder, m.Mobject)
    assert placeholder is not simple_grid[0, 1]


def test_set_and_get_item_back(simple_grid: Grid) -> None:
    c = m.Circle()
    simple_grid[1, 2] = c
    retrieved = simple_grid[1, 2]
    assert retrieved is c


def test_assignment_moves_mobject_to_cell(simple_grid: Grid) -> None:
    sq = m.Square()
    simple_grid[0, 1] = sq
    cell = simple_grid.cells[0][1]
    assert sq.get_center().all() == cell.get_center().all()


def test_assignment_with_alignment_vector(simple_grid: Grid) -> None:
    r = m.Rectangle()
    simple_grid[1, 0, m.UP] = r
    cell = simple_grid.cells[1][0]
    assert r.get_critical_point(m.UP).all() == cell.get_critical_point(m.UP).all()


def test_negative_indices(simple_grid):
    triangle = m.Triangle()
    simple_grid[-1, -1] = triangle
    assert simple_grid[1, 2] is triangle
    assert simple_grid[-1, -1] is triangle


@pytest.mark.parametrize(("row_n", "col_n"), [(5, 0), (0, 5), (-5, 0), (0, -5)])
def test_out_of_range_index_raises(simple_grid: Grid, row_n: int, col_n: int) -> None:
    with pytest.raises(IndexError):
        _ = simple_grid[row_n, col_n]
    with pytest.raises(IndexError):
        simple_grid[row_n, col_n] = m.Dot()


# ----------------------------------------------------------------------
# Margins
# ----------------------------------------------------------------------
def test_margin_is_applied_consistently(simple_grid: Grid) -> None:
    mob1 = m.Square()
    simple_grid[0, 0, m.RIGHT] = mob1
    cell = simple_grid.cells[0][0]
    expected_center = cell.get_center() - m.RIGHT * simple_grid.margin
    assert mob1.get_center().all() == expected_center.all()

    mob2 = m.Circle()
    simple_grid[1, 1, m.DR] = mob2
    cell = simple_grid.cells[1][1]
    expected_center = (
        cell.get_center() - m.DOWN * simple_grid.margin - m.RIGHT * simple_grid.margin
    )
    assert mob2.get_center().all() == expected_center.all()
