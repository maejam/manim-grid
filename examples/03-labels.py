"""Give meaningful names to rows and columns."""

from manim import *

from manim_grid import Grid


class Labels(Scene):
    def construct(self):
        # Give meaningful names with `row_labels` and `col_labels`
        grid = Grid([1] * 5, [1] * 3, col_labels=["row_label", "left", "right"])
        self.add(grid)
        grid.lattice.set_stroke(opacity=1)

        # Add the labels to the grid as Text mobjects
        grid.mobs[0, :] = grid.col_labels(font_size=12)
        # row_labels was not provided => 1-based numeric labels have been auto-generated
        grid.mobs[:, 0] = grid.row_labels(font_size=12, color=BLUE)

        # Labels can be used in-place of numeric indexes for more expressive indexing
        grid.mobs["2", "right"] = Circle(radius=0.2)  # equivalent to grid.mobs[1, 2]
        # Also works for slices and other complex indexes
        grid.rects["2"::2].set_fill(WHITE, opacity=1)
        grid.add(grid.mobs[:])
