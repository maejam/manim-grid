"""Use signals to automate your grid behaviour.

manim-grid provides signals based on blinker that you can use as hooks to add event
callbacks to your scenes.
This provides many possibilities such as triggering a series of actions when tagging
a cell, or removing old mobjects and adding new ones automatically... the possibilities
are endless.

The available signals as well as their documentation can be found in
`src/manim-grid/signals.py`.

The blinker documentation can be found at https://blinker.readthedocs.io/en/stable/
"""

from manim import *

from manim_grid import Grid, mob_inserted, mobs_added


# Functions can be registered easily
@mobs_added.connect
def log_mobs_added(grid, index, mobs):
    logger.info(f"You just added a {len(mobs)} object(s).")


class Signals(Scene):
    def setup(self):
        self.grid = Grid([1] * 5, [1] * 3)

        # Instance methods cannot be registered with the decorator syntax.
        # You can register them inline or make them static (see below)
        mob_inserted.connect(self.rotate_mob_in_cell_2_1)

        # or better yet, connect with a specific sender (here a cell)
        # the callback will only be called for that cell
        mob_inserted.connect(self.paint_blue, sender=self.grid.cells[2, 1])

    def rotate_mob_in_cell_2_1(self, cell, grid):
        if cell.row_index == 2 and cell.col_index == 1:
            cell.mob.rotate(PI)

    def paint_blue(self, cell, grid):
        cell.mob.set_color(BLUE)

    # Alternatively connect a staticmethod with the decorator syntax
    # you loose access to the scene instance (self)
    # Use `@signal_name.connect_via(sender=...)` to provide the sender parameter
    @staticmethod
    @mobs_added.connect
    def add_as_sumobjects(grid, index, mobs):
        grid.add(*grid.mobs[index])

    def construct(self):
        self.add(self.grid)
        self.grid.mobs[:] = [Text(str(n)) for n in range(15)]
        self.wait()
        self.grid.mobs[:, 1] = [Triangle(color=RED).scale(0.7) for _ in range(5)]
        self.wait()
