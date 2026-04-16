"""Inserting rows and columns

For performance and simplicty reasons, the Grid geometry stays fixed. This means that
inserting a new row for example requires pre-allocating one extra row in your Grid.
Otherwise the last row will be lost (which is supported and sometimes desirable).

"""

from manim import *

from manim_grid import Grid, row_insertion_processed


class Inserting(Scene):
    def construct(self):
        grid = Grid(
            [0.6] * 10, [1] * 5, col_labels=["col1", "col2", "col3", "col4", "extra"]
        )

        grid.rects[::2, :-1].set_fill(opacity=0.1)
        # We leave one empty row/col at the end
        # here in RED for demonstration - remove the coloring for better looking animations
        grid.rects[-1].set_fill(RED, opacity=0.5)
        grid.rects[:, -1].set_fill(RED, opacity=0.5)
        grid.mobs[:-1, 0, LEFT] = grid.row_numbers(start=0, stop=9, font_size=18)
        grid.mobs[0, RIGHT] = grid.col_labels(font_size=14)
        grid.mobs[3, 1] = Dot(color=BLUE)
        grid.add(*grid.mobs[:])
        self.add(grid.to_edge(UP))
        self.wait()

        # Since we provided col_labels, we MUST provide a label for the new column.
        # We could also provide a custom `width`, but since our Grid has uniform column
        # widths, the new column width will default to that uniform width (1).
        with grid.insert_column("col2", label="col1.5") as (anim, last_col, tracker):
            # insert_row and insert_column act as context managers. They yield:
            # 1. an animation object that can be played to animate the shift.
            # 2. the last_col (or row) as a VDict with keys "mobs", "olds" and rects.
            # 3. a ValueTracker oject that tracks the advancement of the animation (alpha).
            self.play(anim, run_time=2)

        self.wait()

        with grid.insert_row(3, height=1.2):
            # Inside the context manager, the Grid internal state is already updated
            # (the new row/col is added and the last one is removed), but the visual
            # shift only happens when playing the animation or exiting the context.
            # This gives you a chance to re-style the Grid, including the newly
            # inserted row/column. Since the last row/col is removed and thus inaccesible,
            # last_row/last_col can be used to animate its visual removal (eg FadeOut).

            # Here, the row numbers are now out of sync from the inserted row downward:
            grid.remove(*grid.mobs[3:, 0])
            grid.mobs[3:, 0, LEFT] = grid.row_numbers(3, 10, font_size=18, color=GREEN)
            grid.add(*grid.mobs[3:, 0])

            # The rows opacities need a fix as well
            grid.rects[::2].set_fill(opacity=0.1)
            grid.rects[1::2].set_fill(opacity=0)

        # the animation is not played => an instant shift will happen here
        # moreover, the last row is visually removed if it was not already

        self.wait()


class InsertingWithSignal(Scene):
    """
    This Scene demonstrates a similar example using signals to automate the logic
    when inserting a row.

    The provided signals for rows are:
     - row_insertion_processed: emitted when the internal state of the grid is updated,
       before the visual shift.
     - row_insertion_displayed: emitted when the insertion is complete, including the
       visual shift.
    And similarly for columns: column_insertion_processed and column_insertion_displayed
    See src/manim_grid.signals.py for more details.
    """

    def on_row_inserted(
        self,
        grid,
        row_index,
        height,
        label,
        animation,
        tracker,
        shift_group_factory,
        shift_vec,
        last_row,
    ):
        # update row numbers
        grid.remove(*grid.mobs[row_index:, 0])
        grid.mobs[row_index:, 0, LEFT] = grid.row_numbers(
            row_index, 10, font_size=18, color=GREEN
        )
        grid.add(*grid.mobs[row_index:, 0])

        # fix row opacities
        grid.rects[::2].set_fill(opacity=0.1)
        grid.rects[1::2].set_fill(opacity=0)


    def setup(self):
        row_insertion_processed.connect(self.on_row_inserted)

    def construct(self):
        grid = Grid(
            [0.6] * 10, [1] * 5, col_labels=["col1", "col2", "col3", "col4", "extra"]
        )

        grid.rects[::2, :-1].set_fill(opacity=0.1)
        grid.mobs[:-1, 0, LEFT] = grid.row_numbers(start=0, stop=9, font_size=18)
        grid.mobs[0, RIGHT] = grid.col_labels(font_size=14)
        grid.mobs[3, 1] = Dot(color=BLUE)
        grid.add(*grid.mobs[:])
        self.add(grid.to_edge(UP))
        self.wait()

        with grid.insert_column("col2", label="col1.5"):
            pass

        self.wait()

        # The insertion methods should always be used as context managers even when no
        # styling or animating is required (kind of ugly, I know...)
        with grid.insert_row(3):
            pass

        self.wait()

        # every row insertion is now automatically styled
        # but since we pre-allocated only one extra row, the last row will disappear
        # instead of a row being visually added
        with grid.insert_row(1):
            pass

        self.wait()

        # it is possible to add custom code before the signal handler on a
        # per-insertion basis
        with grid.insert_row(1):
            grid.mobs[3,3] = Dot(color=RED)
            grid.add(grid.mobs[3,3])

        self.wait()
