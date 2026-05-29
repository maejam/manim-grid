from enum import Enum

from manim_utils import LazyAnimation, TrackedAnimationMixin


# constants and sentinels
class _Unset(Enum):
    UNSET = "<UNSET>"


UNSET = _Unset.UNSET
"""Sentinel for default parameters where everything else is valid."""


class _Missing(Enum):
    MISSING = "<MISSING>"

    def __repr__(self) -> str:
        return self.value


MISSING = _Missing.MISSING
"""Sentinel used to signal a missing object."""


class _Deleted:
    DELETED = "<DELETED>"


DELETED = _Deleted.DELETED
"""Sentinel used to signal that an item was deleted."""


# classes
class TrackedLazyAnimation(TrackedAnimationMixin, LazyAnimation): ...
