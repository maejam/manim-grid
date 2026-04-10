import manim as m

from manim_grid.helpers import TrackedApplyMethod


def test_TrackedApplyMethod_correctly_identifies_played_animation():
    anim = TrackedApplyMethod(m.Circle().shift, m.DOWN)
    assert not anim._played
    s = m.Scene()
    s.play(anim)
    assert anim._played
