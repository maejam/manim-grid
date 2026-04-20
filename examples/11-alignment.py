from manim import *

from manim_grid import Grid

class AlignmentExample(Scene):
    def construct(self) -> None:
        grid = Grid([2]*3, [2]*3)
        grid.lattice.set_stroke(opacity=1)
        self.add(grid)

        # Mobjects are centered in the Cells by default
        grid.mobs[:] = [Text("default", font_size=16) for _ in range(9)]

        # The alignement proxy allows to change alignment without assigning
        # It does not move the mobjects. Only changes alignment for next insertions
        # Useful to change default alignment for a column for instance
        grid.alignment[:, -1] = DR
        grid.mobs[:, -1] = [Text("proxy", font_size=16) for _ in range(3)]

        # Alignment can be changed when assigning
        grid.mobs[0, DOWN] = [Text("assigned", font_size=16) for _ in range(3)]

        # Previous alignment is remembered if not explicitly set
        grid.mobs[:, 0] = [Text('remembered', font_size=16) for _ in range(3)]


        grid.add(grid.mobs[:])
