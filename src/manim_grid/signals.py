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
        The Cell instance.
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

tags_replaced = signal(
    "tags_replaced",
    doc="""Emitted when the Tags instance is replaced on  a cell.

    This signal is sent when users call `grid.tags[index] = {...}`, thus replacing
    (as opposed to mutating) the Tags instance(s) on the targeted cell(s). As many
    signals as targeted cells are sent. To act on the whole batch at once, use the
    `is_first_in_batch` and `is_last_in_batch` flags.

    Data
    ----
    sender
        The Cell instance.
    grid
        The Grid instance.
    old_tags_instance
        The replaced Tags instance.
    new_tags_instance
        The Tags instance assigned to the cell.
    index
        The index passed to `grid.tags`. It is either a tuple `(row_key, col_key)` or a
        single `row_key` where row_key and col_key can be anything supported by the
        proxies (int, label, slice, array...). Can be used to index any proxy to resolve
        to the same cells. This signal is sent for each Cell targeted by the `index`.
    is_first_in_batch
        Because many cells can be targeted by the index and one signal is sent for each
        cell, this flag lets you know if this signal is the first sent for this batch of
        cells.
    is_last_in_batch
        Similar to `is_first_in_batch` for the last signal in the batch.

    Return Value
    ------------
    None

    Example
    -------

    See Also
    --------
    tag_mutated
    """,
)

tag_mutated = signal(
    "tag_mutated",
    doc="""Emitted when a tag value is changed or deleted.

    Unlike `tags_replaced`, the Tags instance is not replaced but simply mutated.

    Data
    ----
    sender
        The Cell instance.
    grid
        The Grid instance.
    before
        The Tags instance state before mutation (dict).
    after
        The Tags instance state after mutation (dict).
    key
        The mutated key (str).
    value
        The assigned value. `DELETED` for a key deletion operation.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # add the value as a Text mobject when a given tag key is inserted

    >>> @tag_mutated.connect
    >>> def add_as_Text(cell, grid, before, after, key, value):
    >>>     if key == "special_key":
    >>>         if value is not DELETED:
    >>>             txt = grid.mobs[cell.row_index, cell.col_index] = Text(value)
    >>>             grid.add(txt)
    >>>         else:
    >>>             grid.remove(grid.mobs[cell.row_index, cell.col_index])

    See Also
    --------
    tag_mutated
    """,
)
