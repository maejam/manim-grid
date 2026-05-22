from functools import partial

import manim as m
import pytest

from manim_grid.grid import Cell, Grid
from manim_grid.proxies.cells_updater_proxy import CellUpdater, CellUpdaterList
from manim_grid.proxies.config_proxy import Config

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
grid = Grid([2], [2])


@pytest.fixture
def cell():
    cell = Cell(grid, m.Rectangle(), 0, 1)
    cell.config = Config(cell, a=1, b=2, c=3)
    cell.config.set_priority("a", 1)
    cell.config.set_priority("b", 0)
    cell.config.set_priority("c", -1)
    return cell


@pytest.fixture
def cell2():
    cell = Cell(grid, m.Rectangle(), 1, 1)
    cell.config = Config(cell, a=10, b=20)
    cell.config.set_priority("a", 0)
    cell.config.set_priority("b", 1)
    return cell


@pytest.fixture
def updater(cell):
    updater = CellUpdater(cell)
    return updater


@pytest.fixture
def updater_list(updater, cell2):
    updater2 = CellUpdater(cell2)
    ul = CellUpdaterList([updater, updater2])
    return ul


# ----------------------------------------------------------------------
# CellUpdater
# ----------------------------------------------------------------------
def test_cell_updater_initialization(cell, updater):
    assert updater._owner == cell
    assert updater.name == "CellUpdater[0, 1]"
    assert updater._updater is None
    assert len(updater.updaters) == 0


def test_merge_config_no_keys(updater):
    result = updater._merge_config()
    assert list(result.keys()) == ["c", "b", "a"]
    assert result == {"c": 3, "b": 2, "a": 1}


def test_merge_config_with_keys_not_in_config_raises(updater):
    with pytest.raises(
        KeyError,
        match=r"Cell\(row_index=0, col_index=1\) does not have 'd' config key.",
    ):
        updater._merge_config(["d"])


def test_merge_config_with_keys_respects_order(updater):
    result = updater._merge_config(["a", "c", "b"])
    assert list(result.keys()) == ["a", "c", "b"]
    assert result == {"a": 1, "b": 2, "c": 3}


def test_merge_config_with_keys_subset(updater):
    result = updater._merge_config(["a", "b"])
    assert list(result.keys()) == ["a", "b"]
    assert result == {"a": 1, "b": 2}


def test_merge_config_with_overrides(updater):
    result = updater._merge_config(a=99, c=100)
    assert list(result.keys()) == ["c", "b", "a"]
    assert result == {"c": 100, "b": 2, "a": 99}


def test_merge_config_with_keys_and_overrides(updater):
    result = updater._merge_config(["a", "b"], a=99)
    assert list(result.keys()) == ["a", "b"]
    assert result == {"b": 2, "a": 99}


def test_merge_config_with_keys_and_overrides_and_additional_key(updater):
    result = updater._merge_config(["a", "b"], d=99)
    assert list(result.keys()) == ["a", "b", "d"]
    assert result == {"b": 2, "a": 1, "d": 99}


def test_merge_config_with_keys_and_overrides_keys_determine_order(updater):
    result = updater._merge_config(["a", "b"], b=0, a=99)
    assert list(result.keys()) == ["a", "b"]
    assert result == {"b": 0, "a": 99}


def test_merge_config_with_empty_keys_iterable_and_overrides(updater):
    result = updater._merge_config([], b=0, d=99)
    assert list(result.keys()) == ["b", "d"]
    assert result == {"b": 0, "d": 99}


def test_attach_updaters_creates_partial(updater):
    assert len(updater.updaters) == 0
    updater._attach_updaters(color="blue", opacity=0.5)
    assert isinstance(updater.updaters[0], partial)
    assert len(updater.updaters) == 1


def test_detach_updaters_removes_updater(updater):
    updater._attach_updaters([], color="blue")
    assert isinstance(updater.updaters[0], partial)
    assert len(updater.updaters) == 1
    updater._detach_updaters()
    assert len(updater.updaters) == 0


def test_call_method_triggers_update(updater):
    call_args = []

    def mock_update(self, **merged):
        call_args.append(merged)

    updater._update = mock_update

    # Call the updater directly
    updater([], color="green")

    assert len(call_args) == 1
    assert call_args[0] == {"color": "green"}


def test_run_context_manager(updater):
    assert len(updater.updaters) == 0
    with updater.run("a", color="red"):
        assert len(updater.updaters) == 1

    assert len(updater.updaters) == 0


def test_run_nested_context_managers(updater):
    assert len(updater.updaters) == 0
    with updater.run(color="red"), updater.run(opacity=0.5):
        assert len(updater.updaters) == 2

    assert len(updater.updaters) == 0


def test_run_decorator(updater):
    @updater.run(scale=2.0)
    def my_function(arg, *, kwarg):
        assert len(updater.updaters) == 1
        return f"executed with {arg} and {kwarg=}"

    assert len(updater.updaters) == 0
    result = my_function(1, kwarg="k")
    assert result == "executed with 1 and kwarg='k'"
    assert len(updater.updaters) == 0


def test_run_nested_decorators(updater):
    @updater.run(scale=2.0)
    @updater.run(color="green")
    def my_function(arg, *, kwarg):
        assert len(updater.updaters) == 2
        return f"executed with {arg} and {kwarg=}"

    assert len(updater.updaters) == 0
    result = my_function(1, kwarg="k")
    assert result == "executed with 1 and kwarg='k'"
    assert len(updater.updaters) == 0


def test_cell_updater_list_iteration(updater_list, updater):
    assert len(updater_list) == 2
    assert updater_list[0] is updater
    assert isinstance(updater_list[1], CellUpdater)
    assert updater_list[1] is not updater


def test_cell_updater_list_attach_detach(updater_list):
    updater_list._attach_updaters(opacity=0.5)

    assert len(updater_list[0].updaters) == 1
    assert len(updater_list[1].updaters) == 1
    assert updater_list[0].updaters[0] is not updater_list[1].updaters[0]
    assert updater_list[0].updaters[0].keywords == {
        "c": 3,
        "b": 2,
        "a": 1,
        "opacity": 0.5,
    }
    assert updater_list[1].updaters[0].keywords == {
        "a": 10,
        "b": 20,
        "opacity": 0.5,
    }

    updater_list._detach_updaters()
    assert len(updater_list[0].updaters) == 0
    assert len(updater_list[1].updaters) == 0


def test_context_manager_exception_handling(updater):
    try:
        with updater.run(color="red"):
            assert len(updater.updaters) == 1
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Even with exception, __exit__ should run and detach
    assert len(updater.updaters) == 0


def test_decorator_exception_handling(updater):
    @updater.run(color="red")
    def failing_func():
        raise RuntimeError("Boom")

    with pytest.raises(RuntimeError):
        failing_func()

    # Should be cleaned up despite exception
    assert len(updater.updaters) == 0
