from contextlib import contextmanager
from typing import Any

import manim as m
import pytest
from blinker import signal

from manim_grid import Grid


# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------
@pytest.fixture
def simple_grid():
    """A 2x3 grid."""
    row_heights = [1.0, 1.0]
    col_widths = [1.5, 1.5, 1.5]
    g = Grid(
        row_heights,
        col_widths,
        buff=(0.1, 0.3),
        margin=(0.1, 0.3),
        row_labels=(),
        col_labels=(),
        num_visible_rows=1,
    )
    return g


@pytest.fixture
def dummy_mob():
    class DummyMobject(m.Mobject):
        def __init__(self):
            self.pos = None
            self.aligned_edge = None
            self.shift_vec = None

        def move_to(self, target, aligned_edge=None):  # type:ignore[reportIncompatibleMethodOverride]
            self.pos = target
            self.aligned_edge = aligned_edge
            return self

        def shift(self, vec):  # type:ignore[reportIncompatibleMethodOverride]
            self.shift_vec = vec
            return self

        def __repr__(self):
            return f"<DummyMobject pos={self.pos} aligned={self.aligned_edge}>"

    return DummyMobject()


# ----------------------------------------------------------------------
# Blinker signals
# ----------------------------------------------------------------------
class ReceivedSignal:
    def __init__(self, sender: Any, kwargs: dict):
        self._sender = sender
        self._data = kwargs

    def __getattr__(self, name):
        if name == "sender":
            return self._sender
        return self._data[name]

    def __repr__(self):
        return f"ReceivedSignal({self._sender}, {self._data})"


class SignalMonitor:
    def __init__(self, received: list):
        self._received = received
        self._index = 0

    def __len__(self):
        return len(self._received)

    def __iter__(self):
        return iter(self._received)

    def __next__(self):
        if self._index >= len(self._received):
            raise AssertionError(
                f"No more signals received. Received {len(self._received)}."
            )
        result = self._received[self._index]
        self._index += 1
        return result

    def __getitem__(self, index):
        if index >= len(self._received):
            raise IndexError(
                f"Signal index {index} out of range. Received {len(self._received)}."
            )
        return self._received[index]

    def assert_received(self, count=1):
        assert len(self._received) == count, (
            f"Expected {count} signal(s), got {len(self._received)}"
        )

    def assert_not_received(self):
        assert len(self._received) == 0, (
            f"Expected no signals, got {len(self._received)}"
        )


@pytest.fixture
def signal_monitor():
    @contextmanager
    def monitor(signame: str, weak=False):
        received = []

        def handler(sender, **kwargs):
            received.append(ReceivedSignal(sender, kwargs))

        sig = signal(signame)
        sig.connect(handler, weak=weak)
        try:
            yield SignalMonitor(received)
        finally:
            sig.disconnect(handler)

    return monitor
