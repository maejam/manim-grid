"""Tag cells.

This example shows how to tag cells with custom key/value metadata.
To see how this can reveal very useful, see the examples on 'masking' and 'signals'
"""

from manim import *

from manim_grid import Grid


class Tagging(Scene):
    def construct(self):
        grid = Grid([0.5] * 10, [2] * 2)

        # tagging a single cell
        grid.tags[0, 0].foo = "bar"

        # tagging all cells in the last row at once
        grid.tags[-1].foo = "baz"

        # updating first column tags
        # all dict methods work the same way on multiple cells or on a single cell
        grid.tags[:, 0].update(foo=42)

        # retrieving a single tag for a single cell
        print(grid.tags[0, 0].foo)  # 42
        print(type(grid.tags[0, 0].foo))  # <class 'int'>

        # retrieving all tags for a single cell
        print(grid.tags[0, 0])  # {"foo": 42}
        print(type(grid.tags[0, 0]))  # <class 'manim_grid.proxies.tags_proxy.Tags'>

        # retrieving a single tag for multiple cells
        # missing values are handled gracefully
        print(grid.tags[0].foo)  # [42 <MISSING>]
        print(type(grid.tags[0].foo))  # <class 'list' >

        # retrieving all tags for multiple cells
        print(grid.tags[0])  # [Tags(foo=42), Tags()]
        print(type(grid.tags[0]))  # <class 'manim_grid.proxies.tags_proxy.TagsList'>

        # Replacing all the tags on cell(s). Accepts a `Tags` instance or a Mapping.
        grid.tags[1] = {"baz": False, "foo": 24}

        print(grid.tags)
        # [['Tags(foo=42)' 'Tags()']
        #  ['Tags(baz=False, foo=24)' 'Tags(baz=False, foo=24)']
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
