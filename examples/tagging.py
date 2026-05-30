"""Tag cells.

This example shows how to tag cells with custom key/value metadata.
To see how this can become very useful, see the examples on 'masking' and 'signals'.
"""

from manim import *

from manim_grid import Grid


class Tagging(Scene):
    def construct(self):
        grid = Grid([0.5] * 10, [2] * 2)

        # tagging a single cell
        grid.tags[0, 0].foo = "bar"

        # tagging all cells in the last row at once with a Sequence of values
        grid.tags[-1].foo = [1, 2]
        # when passing a scalar value, it is broadcasted to fill the selection
        # be careful with mutable values
        grid.tags[-1].foo = "baz"  # equivalent to ["baz", "baz"]

        # updating first column tags
        # all dict methods work the same way on multiple cells or on a single cell
        grid.tags[:, 0].update(foo=42)

        # retrieving a single tag for a single cell
        print(grid.tags[0, 0].foo)  # 42
        print(type(grid.tags[0, 0].foo))  # <class 'int'>

        # retrieving all tags for a single cell
        print(grid.tags[0, 0])  # Tags(foo=42)
        print(type(grid.tags[0, 0]))  # <class 'manim_grid.proxies.tags_proxy.Tags'>

        # retrieving a single tag for multiple cells
        # missing values are handled gracefully
        print(grid.tags[0].foo)  # [42 <MISSING>]
        print(type(grid.tags[0].foo))  # <class 'list' >

        # retrieving all tags for multiple cells
        print(grid.tags[0])  # [Tags(foo=42), Tags()]
        print(type(grid.tags[0]))  # <class 'manim_grid.proxies.tags_proxy.TagsList'>

        print(grid.tags)
        # [['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' 'Tags()']
        #  ['Tags(foo=42)' "Tags(foo='baz')"]]
        print(type(grid.tags))
        # <class 'manim_grid.proxies.tags_proxy.TagsProxy'>

        # A Tags instance is also attached to the grid itself and is accessible
        # through the `grid.gtags` attribute (g for grid or global)
        # Just like Tags instances attached to Cells, it supports dot notation as well
        # as key indexing
        grid.gtags.foo = "bar"
        print(grid.gtags["foo"])
        # 'bar'
