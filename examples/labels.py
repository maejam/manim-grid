"""Give meaningful names to rows and columns."""

from manim import *

from manim_grid import Grid


class LabelsAndNumbers(Scene):
    def construct(self):
        # Give meaningful names with `row_labels` and `col_labels`
        grid = Grid(
            [1] * 5, [1] * 4, col_labels=["row\nnumber", "left", "center", "right"]
        )
        self.add(grid)
        grid.lattice.set_stroke(opacity=1)

        # Add the labels to the grid as Text mobjects
        grid.mobs[0, :] = grid.col_labels(font_size=16)
        # row_labels was not provided. We can still add row_numbers.
        # start/stop/step are optional. If not given a list with as many numbers
        # as the number of rows is generated.
        grid.mobs[1:, 0] = grid.row_numbers(
            start=1, stop=5, step=1, font_size=26, color=BLUE
        )
        # the line above is equivalent to:
        # grid.mobs[1:, 0] = grid.row_numbers(font_size=26, color=BLUE)[0:4:1]

        # Labels can be used in-place of numeric indexes for more expressive indexing
        grid.mobs[1, "right"] = Circle(radius=0.2)  # equivalent to grid.mobs[1, 3]
        # Also works for slices and other complex indexes
        grid.rects[:, "left"::2].set_fill(WHITE, opacity=0.2)
        grid.add(*grid.mobs[:])
