from blinker import signal

mobs_assigned = signal(
    "mobs_assigned",
    doc="""Emitted when mobjects are assigned to the Grid.

    This signal is sent when users call `grid.mobs[index] = mob(s)`. Meant to be used
    to act on all assigned mobjects in the Grid indiscriminately.

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
    mobs_added

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
    mobs_assigned
    mobs_added

    """,
)

mob_added = signal(
    "mob_added",
    doc="""Emitted when mobjects are added as submobjects to the Grid.

    Data
    ----
    sender
        The mobject that is added as submobject.
    grid
        The Grid instance.

    Return Value
    ------------
    None

    See Also
    --------
    mob_removed

    """,
)

mob_removed = signal(
    "mob_removed",
    doc="""Emitted when mobjects are removed from the Grid submobjects.

    Data
    ----
    sender
        The mobject that is removed from the Grid submobjects.
    grid
        The Grid instance.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # log a warning when a (V)Group is added or removed

    >>> @mob_added.connect
    >>> @mob_removed.connect
    >>> def warn_on_group(mob, grid):
    >>>     if isinstance(mob, (Group, VGroup)):
    >>>         logger.warning("A Group was added/removed: %s", mob.submobjects)

    See Also
    --------
    mob_added

    """,
)
tag_changed = signal(
    "tag_changed",
    doc="""Emitted when a tag value is changed or deleted.

    Data
    ----
    sender
        The Cell or Grid instance that owns tha Tags.
    grid
        The Grid instance.
    before
        The Tags instance state before mutation (dict).
    after
        The Tags instance state after mutation (dict).
    key
        The new key (str).
    value
        The assigned value. `DELETED` for a key deletion operation.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # add the value as a Text mobject when a given tag key is inserted

    >>> @tag_changed.connect
    >>> def add_as_Text(cell, grid, before, after, key, value):
    >>>     if key == "special_key":
    >>>         if value is not DELETED:
    >>>             txt = grid.mobs[cell.row_index, cell.col_index] = Text(value)
    >>>             grid.add(txt)
    >>>         else:
    >>>             grid.remove(grid.mobs[cell.row_index, cell.col_index])

    """,
)

row_insertion_processed = signal(
    "row_insertion_processed",
    doc="""Emitted when the internal state of the Grid is updated with the new row.

    The visual shift did not happen yet, but the new row is already added to the
    Grid logical state, and the last one is removed from it.

    Data
    ----
    sender
        The Grid instance.
    row_index
        The index of the inserted row.
    height
        The height of the inserted row.
    label
        The string label given to the inserted row. None if not given.
    animation
        The shift animation for the rows after the inserted one.
    tracker
        The ValueTracker tracking the advancement of the animation.
    shift_group_factory
        A function that when called will return a VGroup containing the mobjects to
        shift. Includes the `mobs`/`olds` and `rects` for the rows to shift.
    shift_vec
        The vector by which to shift the shift_group.
    last_row
        The mobjects (`mobs`/`olds` and `rects`) for the last row that are not accesible
        through the proxies anymore.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # interpolate the grid color as the animation progresses
    >>> @row_insertion_processed.connect
    >>> def interpolate_grid_color(
    >>>     grid,
    >>>     row_index,
    >>>     height,
    >>>     label,
    >>>     animation,
    >>>     tracker,
    >>>     shift_group,
    >>>     shift_vec,
    >>>     last_row,
    >>> ):
    >>>     def update_grid_color(mobs: VMobject, alpha):
    >>>         color = interpolate_color(RED, GREEN, tracker.get_value())
    >>>         mobs.set_stroke(color)

    >>>     self.play(
    >>>         animation,
    >>>         UpdateFromAlphaFunc(
    >>>             VGroup(grid.rects[:], last_row["rects"]), update_grid_color
    >>>         ),
    >>>         run_time=3,
    >>>     ),

    See Also
    --------
    row_insertion_displayed
    column_insertion_processed
    column_insertion_displayed

    """,
)

row_insertion_displayed = signal(
    "row_insertion_displayed",
    doc="""Emitted when the row insertion is complete, including the visual shift.

    Data
    ----
    sender
        The Grid instance.
    row_index
        The index of the inserted row.
    height
        The height of the inserted row.
    label
        The string label given to the inserted row. None if not given.
    animation
        The shift animation for the rows after the inserted one.
    tracker
        The ValueTracker tracking the advancement of the animation.
    shift_group
        The group containing the mobjects to shift. Includes the `mobs`/`olds`
        and `rects` for the rows to shift.
    shift_vec
        The vector by which to shift the shift_group.
    last_row
        The mobjects (`mobs`/`olds` and `rects`) for the last row that are not accesible
        through the proxies anymore.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # play the animation backwards after it has finished
    >>> @row_insertion_displayed.connect
    >>> def revert_shift_animation(
    >>>     grid,
    >>>     row_index,
    >>>     height,
    >>>     label,
    >>>     animation,
    >>>     tracker,
    >>>     shift_group,
    >>>     shift_vec,
    >>>     last_row,
    >>> ):
    >>>     reverse_anim = ApplyMethod(shift_group.shift, -shift_vec)
    >>>     self.play(
    >>>         reverse_anim,
    >>>     )

    See Also
    --------
    row_insertion_processed
    column_insertion_processed
    column_insertion_displayed

    """,
)

column_insertion_processed = signal(
    "column_insertion_processed",
    doc="""Emitted when the internal state of the Grid is updated with the new column.

    The visual shift did not happen yet, but the new column is already added to the
    Grid logical state, and the last one is removed from it.

    Data
    ----
    sender
        The Grid instance.
    col_index
        The index of the inserted column.
    width
        The width of the inserted column.
    label
        The string label given to the inserted column. None if not given.
    animation
        The shift animation for the columns after the inserted one.
    tracker
        The ValueTracker tracking the advancement of the animation.
    shift_group_factory
        A function that when called will return a VGroup containing the mobjects to
        shift. Includes the `mobs`/`olds` and `rects` for the rows to shift.
    shift_vec
        The vector by which to shift the shift group.
    last_row
        The mobjects (`mobs`/`olds` and `rects`) for the last row that are not accesible
        through the proxies anymore.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # cancel the visual shift
    >>> # the animations from multiple insertions could then be pooled and played later
    >>> @column_insertion_processed.connect
    >>> def cancel_shift(
    >>>     grid,
    >>>     col_index,
    >>>     width,
    >>>     label,
    >>>     animation,
    >>>     tracker,
    >>>     shift_group,
    >>>     shift_vec,
    >>>     last_row,
    >>> ):
    >>>     # The shift will happen automatically if and only if
    >>>     # its `status` attribute is "not palyed".
    >>>     animation.status = "played"

    See Also
    --------
    row_insertion_processed
    row_insertion_displayed
    column_insertion_displayed

    """,
)

column_insertion_displayed = signal(
    "column_insertion_displayed",
    doc="""Emitted when the column insertion is complete, including the visual shift.

    Data
    ----
    sender
        The Grid instance.
    col_index
        The index of the inserted column.
    width
        The width of the inserted column.
    label
        The string label given to the inserted column. None if not given.
    animation
        The shift animation for the columns after the inserted one.
    tracker
        The ValueTracker tracking the advancement of the animation.
    shift_group
        The group containing the mobjects to shift. Includes the `mobs`/`olds`
        and `rects` for the columns to shift.
    shift_vec
        The vector by which to shift the shift group.
    last_row
        The mobjects (`mobs`/`olds` and `rects`) for the last row that are not accesible
        through the proxies anymore.

    Return Value
    ------------
    None

    Examples
    --------
    >>> # add the label to the new column
    >>> @column_insertion_displayed.connect
    >>> def add_label(
    >>>     self,
    >>>     grid,
    >>>     col_index,
    >>>     width,
    >>>     label,
    >>>     animation,
    >>>     tracker,
    >>>     shift_group,
    >>>     shift_vec,
    >>>     last_col,
    >>> ):
    >>>     grid.mobs[0, col_index] = Text(label)
    >>>     grid.add(grid.mobs[0, col_index])

    See Also
    --------
    row_insertion_processed
    row_insertion_displayed
    column_insertion_processed

    """,
)
