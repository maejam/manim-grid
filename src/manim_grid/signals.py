from blinker import signal

mobs_added = signal(
    "mobs_added",
    doc="""Emitted when mobjects are added to the Grid.

    This signal is sent when users call `grid.mobs[index] = mob(s)`. Meant to be used
    to act on all added mobjects in the Grid indiscriminately.

    Data
    ----
    sender
        The grid instance.
    index
        The index passed to `grid.mobs`. It is a tuple (row_key, col_key) where row_key
        and col_key can be anything supported by the mobs proxy (int, label, slice,
        array...). Because this index is hard to parse, targeting only specific cells
        is not easy task. The `mob_inserted` signal would be more adapted for that use
        case.
    mobs
        The value(s) assigned to the indexed cells. Always a list, even when only one
        mobject is assigned.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # automatically paint all mobjects in RED and add them as submojects

    >>> @mobs_added.connect
    >>> def paint_red_and_add(grid, index, mobs):
    >>>     for mob in mobs:
    >>>         mob.set_color(RED)
    >>>     grid.add(*mobs)

    See Also
    --------
    mob_inserted
    """,
)

mob_inserted = signal(
    "mob_inserted",
    doc="""Emitted when a mobject is inserted into a cell.

    This signal is sent **after** users have called `grid.mobs[...] = ...`, for each
    mobject insertion into a cell. It is meant to be used when it is necessary to target
    only specific cells. Also, since the mobject is inserted at that point, the
    previous mobject is now accessible through the `old` attribute and the mobject is
    positioned which makes it possible to adjust its position.

    Data
    ----
    sender
        The cell instance. The following Cell attributes are accessible:
        `.mob`, `.old`, `.rect`, `.tags`, `.row_index` and `.col_index`.
    grid
        The grid that cell belongs to.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # automatically remove olds and add mobs as submojects in the first column only

    >>> @mob_inserted.connect
    >>> def auto_remove_add(cell, grid):
    >>>     if cell.col_index == 0:
    >>>         grid.remove(cell.old)
    >>>         grid.add(cell.mob)

    See Also
    --------
    mobs_added
    """,
)
