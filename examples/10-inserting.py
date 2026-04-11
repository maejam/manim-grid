"""Inserting rows and columns

For performance and simplicty reasons, the Grid geometry stays fixed. This means that
inserting a new row for example requires pre-allocating one extra row in your Grid.

"""

from manim import *

from manim_grid import Grid


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
        grid.add(*grid.mobs[:, 0])
        grid.mobs[0, RIGHT] = grid.col_labels(font_size=14)
        grid.mobs[3, 1] = Dot(color=BLUE)
        grid.add(*grid.mobs[:])
        self.add(grid.to_edge(UP))
        self.wait()

        # Since we provided col_labels, we MUST provide a label for the new column.
        # We could also provide a custom `width`, but since our Grid has uniform column
        # widths, the new column width will default to that uniform width (1).
        with grid.insert_column("col2", label="col1.5") as anim:
            # insert_row and insert_column act as context managers.
            # They yield an animation object that can be played to animate the shift.
            self.play(anim, run_time=2)

        self.wait()

        with grid.insert_row(3, height=1.2):
            # Inside the context manager, the Grid internal state is already updated
            # with the new row/col, but the visual shift only happens when playing it or
            # exiting the context.
            # This gives you a chance to re-style the Grid, including the newly
            # inserted row/column.
            # Here, the row numbers are now out of sync from the inserted row downward:
            grid.remove(*grid.mobs[3:, 0])
            grid.mobs[3:, 0, LEFT] = grid.row_numbers(3, 10, font_size=18, color=GREEN)
            grid.add(*grid.mobs[3:, 0])

            # The rows opacities need a fix as well
            grid.rects[::2].set_fill(opacity=0.1)
            grid.rects[1::2].set_fill(opacity=0)

        # the animation is not played => an instant shift will happen here

        self.wait()


# NOTE: The insertion methods should always be used as context managers even when no
# styling or animating is required: `with grid.insert_row(3): pass`
