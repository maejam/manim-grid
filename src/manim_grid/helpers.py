from typing import Any

import manim as m


class TrackedApplyMethod(m.ApplyMethod):
    """Subclass of ApplyMethod that tracks playback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._played = False

    def finish(self) -> None:
        super().finish()
        self._played = True
