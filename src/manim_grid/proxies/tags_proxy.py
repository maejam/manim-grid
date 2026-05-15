from .base import ReadableProxy
from .dict_list import _Dict, _DictList


class Tags(_Dict):
    signal_name = "tag_changed"


class TagsList(_DictList): ...


class TagsProxy(ReadableProxy[Tags, TagsList]):
    """Proxy that forwards attribute access to the ``tags`` field of each Cell.

    It returns a Tags or TagsList object so that the user can request a given tag or
    chain ``.update/.remove/.clear``... after an indexing operation.

    Examples
    --------
    >>> from manim_grid import Grid, MISSING
    >>> import numpy as np
    >>> from manim import *
    >>> # Create a simple 2×3 grid
    >>> g = Grid(row_heights=[1, 1], col_widths=[1, 1, 1])

    Basic attribute access
    ----------------------
    >>> # Set a tag on a single cell using attribute syntax
    >>> g.tags[1, 1].foo = "bar"
    >>> g.tags[1, 1].foo
    'bar'

    Bulk assignment and retrieval
    -----------------------------
    >>> # Set the 'foo' tag on all the cells in the first row
    >>> g.tags[0].foo = "bar"
    >>> g.tags[0].foo
    ['bar', 'bar', 'bar']

    >>> # Using the dict methods
    >>> g.tags[0].update(foo="qux", baz=42)
    >>> # Retrieve the ``foo`` flag for the entire grid
    >>> foo_flags = g.tags[:].foo
    >>> isinstance(foo_flags, TagsList)
    True
    >>> len(foo_flags)
    6

    Missing-tag handling
    --------------------
    >>> # Only the first row received the ``baz`` tag
    >>> baz_tag = g.tags[:].baz
    >>> baz[0, 0]                     # present: returns the value
    42
    >>> baz[1, 2]                     # absent: returns the MISSING sentinel
    '<MISSING>'

    NOTE
    ----
    Setting a mutable object as a value will result in a shared object:

    Example::
        >>> grid = Grid([1]*2, [1]*2)
        >>> grid.tags[0, :].mutable = [1, 2]
        >>> grid.tags[0, 0].mutable.append(3)
        >>> grid.tags
        [['Tags(mutable=[1, 2, 3])' 'Tags(mutable=[1, 2, 3])']
        ['Tags(mutable=[1, 2, 3])' 'Tags(mutable=[1, 2, 3])']]


    See Also
    --------
    manim_grid.proxies.mobs_proxy.MobsProxy,
    manim_grid.proxies.olds_proxy.OldsProxy

    """

    _attr = "tags"
    _bulk_container = TagsList
