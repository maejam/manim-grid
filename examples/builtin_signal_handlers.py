"""Manage predefined default signal handlers.

The library comes with a set of built-in handlers that should be useful most of the
time. This example shows how to disable them and how to enable your own set of handlers
when instantitating a new Grid.
These handlers are defined in `manim_grid.handlers`. Specifically, `default_connections`
is a dictionary mapping string names to tuples (signal_name, handler_callable, sender)
that define the connections that are established for any new Grid by default.
"""

from manim import *

from manim_grid import Grid, ANY
import random


class DefaultHandlersExample(Scene):
    def construct(self):
        grid = Grid([1], [1])

        # The HandlerManager for the Grid is accessible through the `handlers` attribute
        # Printing it reveals the managed handlers, those enabled and those disabled,
        # along with their respective <sender>.
        print(grid.handlers)
        # Handlers[
        #    enabled: [
        #       'warn_on_group_added <ANY>',
        #       'warn_on_group_removed <ANY>',
        #       'align_on_mob_inserted <ANY>',
        #       'align_on_cell_updating <align>',
        #       'scale_on_cell_updating <mode>',
        #       'stretch_on_cell_updating <mode>',
        #       'crop_on_cell_updating <mode>'] |
        #   disabled: []]

        # The HandlerManager allows to enable/disable individual handlers by name.
        grid.handlers.disable("align_on_mob_inserted")

        # Or all of them in one command
        grid.handlers.disable_all()

        # The `initial_handlers` parameter allows to instantiate a Grid with your own
        # set of default handlers.
        def random_rect_fill_color(sender, cell, **kwargs):
            color = random.choice([RED, BLUE, GREEN, YELLOW])
            cell.rect.set_fill(color, opacity=0.4)

        grid2 = Grid(
            [1],
            [1],
            initial_handlers={
                "random_color_on_mob_inserted": (
                    "mob_inserted",
                    random_rect_fill_color,
                    ANY,
                )
            },
        )
        print(grid2.handlers)
        # Handlers[enabled: ['random_color_on_mob_inserted <ANY>'] | disabled: []]

        self.add(grid2)
        for _ in range(5):
            grid2.mobs[0, 0] = Dot()
            self.wait(0.5)

        # Handlers don't have to be managed through the HandlerManager but it can be
        # handy to import your `initial_handlers` dictionary from another module if you
        # find yourself defining the same handlers every time you instantiate a new
        # Grid.
