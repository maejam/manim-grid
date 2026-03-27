from manim import *

from manim_grid import Grid


class AlternativeConstructors(Scene):
    """Alternative ways to build grids.

    manim_grid provides alternative ways to build your grid to best match your use case.
    All keyword arguments are available.

    - fullscreen: builds a grid that spans the whole scene frame from a number of rows
    and a number of columns. Best when using the grid as a ruler.
    """

    def construct(self):
        # fullscreen
        fsgrid = Grid.fullscreen(num_rows=3, num_cols=8)
        fsgrid.lattice.set_stroke(opacity=1)
        self.add(fsgrid)
        self.add(
            Rectangle(height=config.frame_height, width=config.frame_width).set_stroke(
                RED
            )
        )
        t = Text("Grid.fullscreen(num_rows=3, num_cols=8)", font_size=28)
        s = SurroundingRectangle(t).set_fill(color=DARK_BLUE, opacity=1)
        self.add(s, t)
