from manim import *

from manim_grid import Grid
from manim_grid.exceptions import GridShapeError


class Scrolling(Scene):
    """Scroll the Grid."""

    def construct(self):
        # When providing num_visible_rows and/or num_visible_cols, it is then possible
        # to scroll the grid vertically and/or horizontally
        # The cells height/width *must* be uniform in the given dimension
        grid = Grid(
            row_heights=[1] * 10,  # uniform => vertical scrolling possible
            col_widths=[1, 2, 3],  # non-uniform => horizontal scrolling impossible
            num_visible_rows=5,
            num_visible_cols=2,
        )
        self.add(grid.to_corner(UL))
        grid.lattice.set_fill(GREEN, opacity=1)

        grid.mobs[:] = [Text(str(n), font_size=12) for n in range(30)]
        grid.add(grid.mobs[:])

        self.play(grid.animate.scroll(DOWN, 3))  # scrolling DOWN: OK

        try:
            self.play(grid.animate.scroll(RIGHT, 1))  # scrolling RIGHT: not OK
        except GridShapeError as e:
            print(e)

        self.wait()

        # Scrolling past the last row/column gives weird artifacts
        # Add empty lines to avoid them
        self.play(grid.animate.scroll(DOWN, 5))
