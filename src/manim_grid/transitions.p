# brainstorming
# pb: the mobject cannot play animations and it would be dirty to pass refernece to scene
# sol0: provide only `auto_remove_add` flag on grid - super easy, not very useful
# sol1: store Animation after g.mobs[...].__setitem__ in grid.last_anim that the scene can then play - kinda useful, a little more complicated
# sol2: store Animations in deque/list attribute for each cell accesible through grid.anims proxy: very flexible - completely different way to think about scene building
# sol3: provide callbacks (blinker?): grid.on_mob_inserted(cell) / grid.on_frame_added / grid.on_grid_scrolled ...
from typing import Protocol
import manim as m
from manim.animation.transform import ReplacementTransform

from .grid import Cell, Grid


class RemoveAddTransition(Protocol):
    def __call__(self, grid: Grid, cell: Cell, **kwargs) -> None: ...


class AnimationTransition(Protocol):
    def __call__(self, grid: Grid, cell: Cell, **kwargs) -> m.Animation: ...


def remove_and_add(grid: Grid, cell: Cell) -> None:
    grid.remove(cell.old)
    grid.add(cell.mob)


def animation_transition(grid: Grid, cell: Cell, animation: m.Animation) -> m.Animation:
    return m.ReplacementTransform


class MScene(Scene):
    def construct(self):
        g = Grid(...)
        g.set_auto_transitions(remove_and_add)
        g.mobs[0, 0] = Circle()  # auto added
        g.set_auto_transitions(animation_transition, animation=ReplacementTransform)
        g.mobs[0, 0] = m.Rectangle()  # animation stored in g.animations DEQUE
        self.play(*g.anims[0, 0])  # play all anims in first cell
        self.play(*g.anims[:])  # play all anims in whole grid
        self.play(g.anims[0, 0].pop())
