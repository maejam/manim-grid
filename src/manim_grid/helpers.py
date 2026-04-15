from manim_utils import LazyAnimation, TrackedAnimationMixin


# constants and sentinels
class _Unset:
    pass


_UNSET = _Unset()
"""A sentinel for default parameters where everything else is valid."""


# classes
class TrackedLazyAnimation(TrackedAnimationMixin, LazyAnimation): ...
