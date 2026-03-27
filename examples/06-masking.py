from manim import *

from manim_grid import Grid


class Masking(Scene):
    """Build boolean masks to easily select cells that satisfy a given condition.

    Proxies (grid.mobs/olds/tags/rects) provide a `mask` method that can be used to
    build a boolean NumPy array. This array can then be used on any proxy to filter
    the cells that satisfies the expressed condition.
    The possibilities are limitless. This example tries to demonstrates some possible
    uses with a contrived example.
    """

    def construct(self):
        grid = Grid([1] * 3, [1] * 3)

        # place circles in the first row, dots in the second, and squares in the third
        grid.mobs[0] = [Circle(radius=0.2) for _ in range(3)]
        grid.mobs[1] = [Dot() for _ in range(3)]
        grid.mobs[2] = [Square(side_length=0.2) for _ in range(3)]

        # color the first column mobjects RED, the second BLUE, and the third GREEN
        grid.mobs[:, 0].set_color(RED)
        grid.mobs[:, 1].set_color(BLUE)
        grid.mobs[:, 2].set_color(GREEN)

        # change mobjects opacity based on their position in the grid
        for i, mob in enumerate(grid.mobs[:]):
            mob.set_opacity(i / len(grid.mobs[:]))

        grid.add(grid.mobs[:])
        self.add(grid)

        # build a boolean mask to select all cells whith a Circle
        # `mask` can take in a predicate function to filter objects
        is_circle = grid.mobs.mask(predicate=lambda mob: isinstance(mob, Circle))
        print(is_circle)  # Dot inherits from Circle
        # [[ True  True  True]
        #  [ True  True  True]
        #  [False False False]]
        print(type(is_circle))
        # <class 'numpy.ndarray'>

        # build a boolean mask to select all cells whith a BLUE Mobject
        # `mask` can also take in key/value pairs to filter objects on their attributes
        # if the attribute is missing, the cell is not selected
        is_blue = grid.mobs.mask(color=BLUE)
        print(is_blue)
        # [[False  True False]
        #  [False  True False]
        #  [False  True False]]

        # if both the predicate and key/value pairs are provided, all conditions must
        # be met for a cell to be selected
        is_green_opacity_gt_point_five = grid.mobs.mask(
            predicate=lambda mob: mob.get_fill_opacity() > 0.5, color=GREEN
        )

        print(is_green_opacity_gt_point_five)
        # [[False False False]
        #  [False False  True]
        #  [False False  True]]

        # tag all cells containing a blue circle combining 2 masks
        grid.tags[is_circle & is_blue].mob = "blue_circle"

        # later, paint those tagged cells rectangles WHITE building a mask on the fly
        grid.rects[grid.tags[:].mob == "blue_circle"].set_fill(WHITE, opacity=1)
