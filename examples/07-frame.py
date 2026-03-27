from manim import *

from manim_grid import Grid


class Frame(Scene):
    """Adding a frame to your grid."""

    def construct(self):
        grid = Grid([0.5] * 5, [1] * 2, num_visible_rows=2)

        # the viewport surrounds the visible part of the grid
        grid.viewport.set_stroke(YELLOW, opacity=1)

        # start by positioning a mobject that will serve as a frame
        grid.frame = (
            SurroundingRectangle(grid.viewport, buff=1)
            .set_fill(RED, opacity=0.6)
            .set_stroke(opacity=0)
        )
        self.add(grid)
        self.wait()

        # removing the frame: simply set to None
        grid.frame = None
        self.wait()

        # use a blue circle as the grid frame
        grid.frame = Circle().set_fill(BLUE, opacity=1).surround(grid.viewport)
        self.wait()
