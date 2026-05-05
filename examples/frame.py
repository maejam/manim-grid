"""Adding a frame to your grid.

Because the Grid can be partly covered by a `Stencil`, adding a frame around your Grid
is not so trivial. This example demonstrates the built-in `frame` functionality.
"""

from manim import *

from manim_grid import Grid


class Frame(Scene):
    def construct(self):
        grid = Grid([0.5] * 5, [1] * 2, num_visible_rows=2)

        # the viewport surrounds the visible part of the grid
        grid.lattice.set_stroke(opacity=1)
        grid.viewport.set_stroke(YELLOW, opacity=1)

        # start by positioning a mobject that will serve as a frame
        # It does not have to be centered on the viewport
        grid.frame = (
            SurroundingRectangle(grid.viewport, buff=1)
            .set_fill(RED, opacity=0.6)
            .set_stroke(opacity=0)
            .shift(UP * 0.4)
        )

        self.add(grid)
        self.wait()

        # removing the frame: simply set to None
        grid.frame = None
        self.wait()

        # use a blue circle as the grid frame
        grid.frame = (
            Circle()
            .set_fill(BLUE, opacity=0.6)
            .set_stroke(opacity=0, width=40)
            .surround(grid.viewport)
        )
        self.wait()

        # dynamically resizing the viewport stroke width or the grid content
        # should be done in the update_viewport context manager
        # the frame will adjust accordingly
        with grid.update_viewport():
            self.play(grid.viewport.animate.set_stroke(width=50))
            self.play(grid.lattice.set_z_index(-1).animate.scale(0.5))

        # the choice was made to keep the frame margins fixed, like resizing a window
        # in an OS does not change the height of the title bar. It may not always look
        # good with Circles or Stars, but does with Rectangles.
        self.wait()
