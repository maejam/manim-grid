import manim as m
import numpy as np
import pytest

from manim_grid.grid import Cell, EmptyMobject
from manim_grid.proxies.base import MISSING
from manim_grid.tags import DELETED


# ----------------------------------------------------------------------
# mobs_added
# ----------------------------------------------------------------------
def test_mobs_added_scalar(simple_grid, signal_monitor):
    with signal_monitor("mobs_added") as monitor:
        d = m.Dot()
        simple_grid.mobs[0, 0] = d
        event = next(monitor)
        assert event.sender is simple_grid
        assert event.index == (0, 0)
        assert event.mobs == [d]
        monitor.assert_received(1)
        monitor.assert_others("mob_inserted", 1)


def test_mobs_added_bulk(simple_grid, signal_monitor):
    with signal_monitor("mobs_added") as monitor:
        d1 = m.Dot()
        d2 = m.Dot()
        d3 = m.Dot()
        simple_grid.mobs[0] = [d1, d2, d3]
        event = next(monitor)
        assert event.sender is simple_grid
        assert event.index == 0
        assert event.mobs == [d1, d2, d3]
        assert simple_grid.mobs[0, 0] is d1
        assert isinstance(simple_grid.mobs[1, 0], EmptyMobject)
        monitor.assert_received(1)
        monitor.assert_others("mob_inserted", 3)


# ----------------------------------------------------------------------
# mob_inserted
# ----------------------------------------------------------------------
def test_mob_inserted_scalar(simple_grid, signal_monitor):
    with signal_monitor("mob_inserted") as monitor:
        d = m.Dot()
        simple_grid.mobs[0, 0] = d
        event = next(monitor)
        assert isinstance(event.sender, Cell)
        assert event.grid is simple_grid
        monitor.assert_received(1)
        monitor.assert_others("mobs_added", 1)


def test_mob_inserted_bulk(simple_grid, signal_monitor):
    with signal_monitor("mob_inserted") as monitor:
        d1 = m.Dot()
        d2 = m.Dot()
        d3 = m.Dot()
        simple_grid.mobs[0] = [d1, d2, d3]
        event = next(monitor)
        assert isinstance(event.sender, Cell)
        assert event.grid is simple_grid
        assert simple_grid.mobs[0, 0] is d1
        assert isinstance(simple_grid.mobs[1, 0], EmptyMobject)
        monitor.assert_received(3)
        monitor.assert_others("mobs_added", 1)


# ----------------------------------------------------------------------
# tag_changed
# ----------------------------------------------------------------------
def test_tag_changed_scalar_setattr_delattr(simple_grid, signal_monitor):
    with signal_monitor("tag_changed") as monitor:
        simple_grid.tags[0, 0].one = 1
        simple_grid.tags[0, 0].two = 2
        event = next(monitor)
        event = next(monitor)
        assert event.sender is simple_grid.cells[0, 0]
        assert event.before == {"one": 1}
        assert event.after == {"one": 1, "two": 2}
        assert event.key == "two"
        assert event.value == 2
        monitor.assert_received(2)
        monitor.assert_no_others()

        del simple_grid.tags[0, 0].one
        event = next(monitor)
        assert event.sender is simple_grid.cells[0, 0]
        assert event.before == {"one": 1, "two": 2}
        assert event.after == {"two": 2}
        assert event.key == "one"
        assert event.value == DELETED
        monitor.assert_received(3)
        monitor.assert_no_others()


def test_tag_changed_bulk_setattr_delattr(simple_grid, signal_monitor):
    with signal_monitor("tag_changed") as monitor:
        simple_grid.tags[:, 0].one = 1
        event = next(monitor)
        assert event.sender is simple_grid.cells[0, 0]
        assert event.before == {}
        assert event.after == {"one": 1}
        assert event.key == "one"
        assert event.value == 1
        event = next(monitor)
        assert event.sender is simple_grid.cells[1, 0]
        assert event.before == {}
        assert event.after == {"one": 1}
        assert event.key == "one"
        assert event.value == 1
        monitor.assert_received(2)
        monitor.assert_no_others()

        del simple_grid.tags[0].one
        event = next(monitor)
        assert event.sender is simple_grid.cells[0, 0]
        assert event.before == {"one": 1}
        assert event.after == {}
        assert event.key == "one"
        assert event.value is DELETED
        monitor.assert_received(3)
        monitor.assert_no_others()


def test_tag_changed_bulk_update(simple_grid, signal_monitor):
    with signal_monitor("tag_changed") as monitor:
        simple_grid.tags[:, 0].update(one=1, two=2)
        event = next(monitor)
        assert event.sender is simple_grid.cells[0, 0]
        assert event.before == {}
        assert event.after == {"one": 1}
        assert event.key == "one"
        assert event.value == 1
        event = next(monitor)
        assert event.sender is simple_grid.cells[0, 0]
        assert event.before == {"one": 1}
        assert event.after == {"one": 1, "two": 2}
        assert event.key == "two"
        assert event.value == 2

        event = next(monitor)
        assert event.sender is simple_grid.cells[1, 0]
        assert event.before == {}
        assert event.after == {"one": 1}
        assert event.key == "one"
        assert event.value == 1
        event = next(monitor)
        assert event.sender is simple_grid.cells[1, 0]
        assert event.before == {"one": 1}
        assert event.after == {"one": 1, "two": 2}
        assert event.key == "two"
        assert event.value == 2
        monitor.assert_received(4)
        monitor.assert_no_others()


def test_tag_changed_bulk_pop(simple_grid, signal_monitor):
    with signal_monitor("tag_changed") as monitor:
        simple_grid.tags[0, 0].update(one=1, two=2)  # 2
        event = next(monitor)
        event = next(monitor)
        assert simple_grid.tags[0, 0] == {"one": 1, "two": 2}
        monitor.assert_received(2)

        # no default, missing keys
        with pytest.raises(KeyError, match="'one'"):
            _ = simple_grid.tags[0].pop("one")
        # no additional event means atomicity is respected
        monitor.assert_received(2)
        # make sure (0, 0) has not been mutated
        assert simple_grid.tags[0, 0] == {"one": 1, "two": 2}

        # default, missing keys
        popped = simple_grid.tags[0].pop("one", MISSING)  # 1
        assert popped == [1, MISSING, MISSING]
        monitor.assert_received(3)
        event = next(monitor)
        assert event.sender is simple_grid.cells[0, 0]
        assert event.before == {"one": 1, "two": 2}
        assert event.after == {"two": 2}
        assert event.key == "one"
        assert event.value == DELETED
        assert simple_grid.tags[0, 0] == {"two": 2}
        assert simple_grid.tags[0, 1] == {}
        monitor.assert_no_others()


def test_tag_changed_bulk_popitem(simple_grid, signal_monitor):
    with signal_monitor("tag_changed") as monitor:
        simple_grid.tags[0, :2].update(one=1, two=2)  # 4
        event = next(monitor)
        event = next(monitor)
        assert simple_grid.tags[0, 0] == {"one": 1, "two": 2}
        assert simple_grid.tags[0, 1] == {"one": 1, "two": 2}
        assert simple_grid.tags[0, 2] == {}
        monitor.assert_received(4)

        # popitem with empty dict raises
        with pytest.raises(KeyError, match=r"popitem\(\): at least one empty"):
            _ = simple_grid.tags[0].popitem()
        # no additional event means atomicity is respected
        monitor.assert_received(4)
        # make sure (0, 0) has not been mutated and (0, 2) still empty dict
        assert simple_grid.tags[0, 0] == {"one": 1, "two": 2}
        assert simple_grid.tags[0, 2] == {}

        # popitem no empty dict
        simple_grid.tags[0, 0].three = 3  # 1
        monitor.assert_received(5)
        popped = simple_grid.tags[0, :2].popitem()  # 2
        monitor.assert_received(7)
        assert simple_grid.tags[0, 0] == {"one": 1, "two": 2}
        assert simple_grid.tags[0, 1] == {"one": 1}
        assert simple_grid.tags[0, 2] == {}
        event = monitor[-1]
        assert event.sender == simple_grid.cells[0, 1]
        assert event.before == {"one": 1, "two": 2}
        assert event.after == {"one": 1}
        assert event.key == "two"
        assert event.value == DELETED
        assert popped == [("three", 3), ("two", 2)]
        monitor.assert_no_others()


def test_tag_changed_bulk_clear(simple_grid, signal_monitor):
    with signal_monitor("tag_changed") as monitor:
        simple_grid.tags[0, :2].update(one=1, two=2)  # 4
        simple_grid.tags[0].clear()  # 4
        monitor.assert_received(8)
        assert simple_grid.tags[0, 0] == {}
        event = monitor[-1]
        assert event.sender is simple_grid.cells[0, 1]
        assert event.before == {"one": 1}  # clears from the end using Tags.popitem
        assert event.after == {}
        assert event.key == "one"
        assert event.value == DELETED
        monitor.assert_no_others()


def test_tag_changed_bulk_setdefault(simple_grid, signal_monitor):
    with signal_monitor("tag_changed") as monitor:
        simple_grid.tags[0, 0].update(one=1, two=2)  # 2
        simple_grid.tags[0].setdefault("one", 0)  # 2
        monitor.assert_received(4)
        event = monitor[-1]
        assert event.sender == simple_grid.cells[0, -1]
        assert event.before == {}
        assert event.after == {"one": 0}
        assert event.key == "one"
        assert event.value == 0
        assert simple_grid.tags[0, 0] == {"one": 1, "two": 2}
        assert simple_grid.tags[0, 1] == {"one": 0}


# ----------------------------------------------------------------------
# row_insertion
# ----------------------------------------------------------------------
def test_row_insertion(simple_grid, signal_monitor):
    with (
        signal_monitor("row_insertion_processed") as monitor1,
        signal_monitor("row_insertion_displayed") as monitor2,
    ):
        with simple_grid.insert_row(0, height=2) as (anim, last_row, tracker):
            pass
        monitor1.assert_received(1)
        monitor2.assert_received(1)
        assert len(monitor1) == len(monitor2) == 1
        event1 = next(monitor1)
        event2 = next(monitor2)
        assert event1.sender is event2.sender is simple_grid
        assert event1.row_index == event2.row_index == 0
        assert event1.height == event2.height == 2
        assert event1.label is event2.label is None
        assert event1.animation is event2.animation is anim
        assert event1.tracker is event2.tracker is anim.alpha_tracker is tracker
        zipped = zip(
            event1.shift_group_factory(),
            event2.shift_group,
            anim._mobject_factory(),
            strict=True,
        )
        for mob1, mob2, mob3 in zipped:
            assert mob1 is mob2 is mob3
        assert all(type(z) is m.VGroup for z in zipped)
        assert np.array_equal(event1.shift_vec, event2.shift_vec)
        zipped = zip(event1.last_row, event2.last_row, last_row, strict=True)
        for grp1, grp2, grp3 in zipped:
            for mob1, mob2, mob3 in zip(grp1, grp2, grp3, strict=True):
                assert mob1 is mob2 is mob3


# ----------------------------------------------------------------------
# column_insertion
# ----------------------------------------------------------------------
def test_col_insertion(simple_grid, signal_monitor):
    with (
        signal_monitor("column_insertion_processed") as monitor1,
        signal_monitor("column_insertion_displayed") as monitor2,
    ):
        with simple_grid.insert_column(0, width=2) as (anim, last_col, tracker):
            pass
        monitor1.assert_received(1)
        monitor2.assert_received(1)
        assert len(monitor1) == len(monitor2) == 1
        event1 = next(monitor1)
        event2 = next(monitor2)
        assert event1.sender is event2.sender is simple_grid
        assert event1.col_index == event2.col_index == 0
        assert event1.width == event2.width == 2
        assert event1.label is event2.label is None
        assert event1.animation is event2.animation is anim
        assert event1.tracker is event2.tracker is anim.alpha_tracker is tracker
        zipped = zip(
            event1.shift_group_factory(),
            event2.shift_group,
            anim._mobject_factory(),
            strict=True,
        )
        for mob1, mob2, mob3 in zipped:
            assert mob1 is mob2 is mob3
        assert all(type(z) is m.VGroup for z in zipped)
        assert np.array_equal(event1.shift_vec, event2.shift_vec)
        zipped = zip(event1.last_col, event2.last_col, last_col, strict=True)
        for grp1, grp2, grp3 in zipped:
            for mob1, mob2, mob3 in zip(grp1, grp2, grp3, strict=True):
                assert mob1 is mob2 is mob3
