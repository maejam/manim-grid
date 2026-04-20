"""Getting started with manim-grid."""

from manim import *

from manim_grid import Grid


class GettingStarted(Scene):
    def construct(self):
        # Create a 2×3 grid (rows, columns)
        grid = Grid(
            row_heights=[2, 2],
            col_widths=[2, 2, 2],
        )
        self.add(grid)
        # Show the lattice of Rectangles making the grid cells
        grid.lattice.set_stroke(opacity=1)

        # Place mobjects in the top row, aligned to the upper edge.
        # The Mobjects are deliberatly not added to the scene (nor are the previous
        # occupants of those cells, if any, removed) to allow for greater control
        # over animations.
        grid.mobs[0, :, UP] = [
            Circle(radius=0.5, color=BLUE),
            Dot(color=GREEN),
            Rectangle(height=0.3, width=0.5),
        ]
        grid.add(*grid.mobs[0])

        # Place a square in the top-left cell, centered (default)
        # It replaces the Circle in that cell
        # The Circle is still accessible via grid.olds[0, 0]
        grid.mobs[0, 0] = Square(side_length=0.5, color=RED)

        # Transform the circle into the square.
        self.play(ReplacementTransform(grid.olds[0, 0], grid.mobs[0, 0]))
