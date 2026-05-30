"""Configure and update cells.

Each cell holds a `Config` instance that works like `Tags`: a dict-like object with dot
attribute access that make it possible to interact with a single instance (Config) or
multiple instances (ConfigList) with a similar API.
Cells also hold a `CellUpdater` instance that allows to apply the configuration options
for each cell. It can be used in 3 different ways: one-time call, context manager and
decorator.
As of v0.5, 2 config options are built-in:
* `align`: the alignment vector for mobjects in each cell (first example).
* `mode`: transformation applied to mobjects bigger than their cell rectangle
  (second example).

It is possible to extend the functionality with your own values for those 2 options as
well as with your own config keys (third example).
"""

from manim import *

from manim_grid import Grid, cell_updating


class AlignmentExample(Scene):
    """Set and apply the alignment vector for each cell.

    Because alignment is often used and is not expensive, it is different from other
    config options in 2 ways:
    1. It can be set via a shortcut when assigning.
    2. It is automatically applied to newly inserted mobjects (this behaviour can be
       disabled: see the example on the built-in signal handlers).
    """

    def construct(self) -> None:
        grid = Grid([2] * 3, [2] * 3)
        grid.lattice.set_stroke(opacity=1)
        self.add(grid)

        # Mobjects are centered in the Cells by default
        grid.mobs[:] = [Text("default", font_size=16) for _ in range(9)]

        # Unilke other options, the `align` config option can be set via a shortcut when
        # assigning to cell(s).
        grid.mobs[0, DOWN] = [Text("assigned", font_size=16) for _ in range(3)]

        # The `align` Config option of the ConfigProxy allows to change alignment
        # without assigning.
        # It does not move the mobjects. It only changes alignment for next insertions
        # or for when the updater is called.
        # Useful to change default alignment for a whole column for instance.
        grid.config[:, -1].align = DR
        grid.mobs[:, -1] = [Text("proxy", font_size=16) for _ in range(3)]

        # Previous alignment is remembered
        grid.mobs[:, 0] = [Text("remembered", font_size=16) for _ in range(3)]

        grid.add(grid.mobs[:])


class ModeExample(Scene):
    """Set and apply the mode for each cell.

    Three modes are built-in:
    1. SCALE: scales the mobject to fit inside its rectangle IF it is larger than it.
    2. STRETCH: stretches the mobject to fit inside its rectangle IF it is larger than it.
    3. CROP: crops the mobject to fit inside its rectangle IF it is larger than it.
    Because these are destructive operations, the original mobject is stored in the
    `_mobcopy` attribute on the mobject itself. The mobject can be reset as it was
    originally defined by calling `mob.reset_mob` (see below).
    """

    def construct(self) -> None:
        grid = Grid([1] * 3, [3])
        grid.lattice.set_stroke(opacity=1)
        self.add(grid)

        # By default, the mode is set to "NONE" which does nothing
        grid.mobs[:] = [
            Text("This text is wider than the Cell Rectangle.") for _ in range(3)
        ]
        grid.add(*grid.mobs[:])
        self.wait()

        # Set each cell to a different mode
        grid.config[:].mode = ["SCALE", "STRETCH", "CROP"]
        # It is necessary to call the CellUpdaters for the change to take effect
        # Calling without parameters will apply ALL config options to the selected cells
        grid.update_cells[:]()
        # The order of operation is determined by the `priority` value for each config
        # option, for each cell (lower values applied first).
        print(grid.config[:].get_priority("mode"))  # [0, 0, 0]
        print(grid.config[:].get_priority("align"))  # [100, 100, 100]
        # To update `mode` after `align` for the last row for instance:
        grid.config[2].set_priority("mode", 200)
        print(grid.config[:].get_priority("mode"))  # [0, 0, 200]
        self.wait()

        # It is also possible to set the keys to update, their order  and their values
        # for each call. The `keys` parameter acts as a filter and sets order, while
        # keyword arguments override values. These changes are just for this call and
        # are NOT set in the Config.
        # First, reset the mobjects
        for mob in grid.mobs[:]:
            mob.reset_mob()
        # Apply only the "mode" and set all cells to "SCALE" for this call
        grid.update_cells[:](keys=["mode"], mode="SCALE")
        self.wait()

        # The updater can also be used as a context manager or a decorator to
        # dynamically update the mobjects. It is then necessary to call the `run`
        # method which can also take `keys` and `overrides` as inputs.
        for mob in grid.mobs[:]:
            mob.reset_mob()
        with grid.update_cells[:].run():
            self.play(grid.rects[:].animate.stretch_to_fit_width(14))
        self.wait()

        @grid.update_cells[:].run()
        def squeeze_rects():
            self.play(grid.rects[:].animate.stretch_to_fit_height(0.5), run_time=3)

        squeeze_rects()
        self.wait()


class CustomOptionsExample(Scene):
    """Add custom config keys/values.

    The UpdaterProxy is based on signals: for each cell and for each config key/value
    pair, the `cell_updating` signal is emitted. Defining custom config options simply
    requires defining a signal handler.
    The built-in handlers seen in the previous examples can be found in
    `manim_grid.handlers`.
    """

    def construct(self) -> None:
        grid = Grid([1] * 3, [3])
        grid.lattice.set_stroke(opacity=1)
        self.add(grid)

        # Define the handler - the sender is the config key
        @cell_updating.connect_via("formatting")
        def capitalize_text(sender, key, value, grid, cell):
            if not isinstance(cell.mob, Text) or value != "CAPS":
                return
            result = Text(cell.mob.text.capitalize())
            cell.mob.become(result)

        # Set the config on the cells
        grid.config[0, 0].formatting = "CAPS"

        grid.mobs[:] = [Text("abcd") for _ in range(3)]
        grid.add(*grid.mobs)
        self.wait()

        # Call the updater
        grid.update_cells[:]()
        self.wait()
