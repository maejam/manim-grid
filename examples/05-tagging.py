"""Tag cells.

This example shows how to tag cells with custom key/value metadata.
To see how this can reveal very useful, see the next example about `masking`.
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

        # updating first column tags - similar to dict.update
        grid.tags[:, 0].update(foo=42)

        # retrieving a single tag for a single cell
        print(grid.tags[0, 0].foo)  # 42
        print(type(grid.tags[0, 0].foo))  # <class 'int'>

        # retrieving all tags for a single cell
        print(grid.tags[0, 0])  # {"foo": 42}
        print(
            type(grid.tags[0, 0])
        )  # <class 'manim_grid.proxies.tags_proxy.ScalarTagsSelection'>

        # retrieving a single tag for multiple cells
        # missing values are handled gracefully
        print(grid.tags[0].foo)  # [42 <MISSING>]
        print(
            type(grid.tags[0].foo)
        )  # <class 'manim_grid.proxies.tags_proxy.BulkTagsSelection'>

        # retrieving all tags for multiple cells
        print(grid.tags[0])  # ["{'foo': 42}" '{}']
        print(
            type(grid.tags[0])
        )  # <class 'manim_grid.proxies.tags_proxy.BulkTagsSelection'>

        # the following methods are self explanatory and work on single cells or in bulk
        grid.tags[0, 0].remove("foo")
        grid.tags[0].clear()

        # Replacing all the tags on cell(s). Accepts a `Tags` instance or a Mapping.
        grid.tags[1] = {"baz": False, "foo": 24}

        print(grid.tags)
        #        [['{}' '{}']
        # ["{'baz': False, 'foo': 24}" "{'baz': False, 'foo': 24}"]
        # ["{'foo': 42}" '{}']
        # ["{'foo': 42}" '{}']
        # ["{'foo': 42}" '{}']
        # ["{'foo': 42}" '{}']
        # ["{'foo': 42}" '{}']
        # ["{'foo': 42}" '{}']
        # ["{'foo': 42}" '{}']
        # ["{'foo': 42}" "{'foo': 'baz'}"]]
        print(type(grid.tags))
        # <class 'manim_grid.proxies.tags_proxy.TagsProxy'>
