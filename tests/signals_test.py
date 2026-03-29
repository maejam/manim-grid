import manim as m

from manim_grid.grid import Cell


def test_mobs_added(simple_grid, signal_monitor):
    with signal_monitor("mobs_added") as monitor:
        d = m.Dot()
        simple_grid.mobs[0, 0] = d

        monitor.assert_received(1)
        event = next(monitor)
        assert event.sender is simple_grid
        assert event.index == (0, 0)
        assert event.mobs == [d]


def test_mob_inserted(simple_grid, signal_monitor):
    with signal_monitor("mob_inserted") as monitor:
        d = m.Dot()
        simple_grid.mobs[0, 0] = d

        monitor.assert_received(1)
        event = next(monitor)
        assert isinstance(event._sender, Cell)
        assert event.grid is simple_grid
