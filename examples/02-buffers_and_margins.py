"""Control cells and mobjects positioning."""

from manim import *

from manim_grid import Grid


class BuffersMargins(Scene):
    def construct(self):
        # `buff` controls the spacing between cells
        # `margin` controls the padding inside the cells

        # They can be scalar values: the same value is used horizontally and vertically
        grid1 = Grid(
            row_heights=[1] * 2,
            col_widths=[2] * 2,
            buff=0.5,
            margin=0.3,
        )
        grid1.lattice.set_stroke(RED, opacity=1)
        self.add(grid1.to_corner(UL))

        # The Dot is placed in the upper-left corner of the first cell
        # with a 0.3 margin in both dimensions
        grid1.mobs[0, 0, UL] = Dot()
        grid1.add(*grid1.mobs[:])

        # They can also be 2-tuples (horizontal, vertical)
        grid2 = Grid(
            row_heights=[1] * 2,
            col_widths=[2] * 2,
            buff=(0.1, 1),
            margin=(0.5, 0.0),
        )
        grid2.lattice.set_stroke(GREEN, opacity=1)
        self.add(grid2.next_to(grid1))

        # The Dot is placed in the upper-left corner of the first cell
        # with a 0.5 horizontal margin and no vertical margin
        grid2.mobs[0, 0, UL] = Dot()
        grid2.add(*grid2.mobs[:])
