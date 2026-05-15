from manim_utils import LazyAnimation, TrackedAnimationMixin


# constants and sentinels
class _Unset:
    __slots__ = ()


UNSET = _Unset()
"""Sentinel for default parameters where everything else is valid."""


class _MissingSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<MISSING>"


MISSING = _MissingSentinel()
"""Sentinel used to signal a missing object."""


class _DeletedSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<DELETED>"


DELETED = _DeletedSentinel()
"""Sentinel used to signal that an item was deleted."""


# classes
class TrackedLazyAnimation(TrackedAnimationMixin, LazyAnimation): ...
