# 📐 `manim-grid` - A Simple Grid Container to ease Mobjects positioning and referencing  

`manim-grid` is a lightweight [Manim](https://www.manim.community/) plugin that creates a rectangular grid of cells. It lets you place any `Mobject` into a cell by using natural indexing and handles alignment, margins, and automatic positioning for you. As it is based on [NumPy](https://numpy.org/), the full power of NumPy indexing is made available.  

> **Why this plugin?**  
> It is born from an attempt to build a better `Code` Mobject, more flexible and easier to work with. This goal quickly turned into a flexible multi-purpose tool.

> **How does it differ from existing manim tools**
> Manim’s built‑in `arrange_in_grid` arranges submojects but it doesn’t give you a persistent “cell” abstraction you can address later.
> The `Table` mobject is great but lacks the dynamic behaviour I needed and is mostly a visual tool.
> The `Paragraph` and `Code` mobjects suffer from the same lack of dynamism and flexibility.

`Grid` fills that gap: it is a flexible, polyvalent and dynamic tool that can be used as a "ruler" to position mobjects in a scene, retrieve them later, and re-assign them on the fly. It does not replace the tools mentioned above but rather offers an alternative way to build and think about your scenes.

---  

## Table of Contents  

- [Features](#features)  
- [Installation](#installation)  
- [Getting Started](#getting-started)  
- [More Examples](#more-examples)
- [Internals](#internals)  
  - [Cell](#cell)  
  - [Proxies (`mobs`, `olds`, `rects`, `tags`)](#proxies)  
  - [Tags](#tags)
  - [Stencil Viewport and Scrolling](#stencil-viewport-and-scrolling)

---  

## Features  

- **Declarative geometry** - specify per-row heights and per-column widths.  
- **Automatic layout** - cells are arranged with `arrange_in_grid`.  
- **Margin & buffer control** - fine-tune spacing between cells and padding inside cells.  
- **Pythonic NumPy-based indexing** - the full power of NumPy indexing is supported: negative indices, slices, masks...
- **String labels** - string identifiers can be added to rows and columns to make indexing even more expressive.  
- **Alignment vectors** - align `Mobjects` to any edge/corner in cells.  
- **Powerful management of cell attributes** - access any attribute for a single cell or in bulk transparently.  
- **Per-cell metadata** - add key/value tags to cells, individually or in bulk.  
- **Scrolling** - the `scroll` method lets you scroll the grid in any direction with a smooth animation.
- **Frame** - add a custom frame around the grid that plays well with scrolling.
- **Fully typed** - for better library and end-user code quality.  

---  

## Installation  

For now there is no Pypi package. Install by adding to your `manim` project:  
- create the project if necessary:  
```bash
uv init myproject
cd myproject
```
- add the plugin to your newly created or existing project:  
```bash
uv add git+https://github.com/maejam/manim-grid.git
```
Requires `Python >= 3.11, < 3.14` and `manim >= 0.19`  

---  

## Getting Started  

> [!NOTE]
> It is necessary to be familiar with (NumPy indexing)[https://numpy.org/doc/stable/user/basics.indexing.html] to fully benefit from this plugin.  

```python
from manim import *
from manim_grid import Grid


class GettingStarted(Scene):
    def construct(self):
        # Create a 2×3 grid (rows, columns)
        grid = Grid(
            row_heights=[2, 2],
            col_widths=[2, 2, 2],
        )
        self.add(grid)
        # Show the lattice of Rectangles making the grid cells
        grid.lattice.set_stroke(opacity=1)

        # Place mobjects in the top row, aligned to the upper edge
        # The Mobjects are deliberatly not added to the scene (nor are the previous
        # occupants of those cells, if any, removed) to allow for greater control
        # over animations.
        grid.mobs[0, :, UP] = [
            Circle(radius=0.5, color=BLUE),
            Dot(color=GREEN),
            Rectangle(height=0.3, width=0.5),
        ]
        grid.add(grid.mobs[0])

        # Place a square in the top-left cell, centered (default)
        # It replaces the Circle in that cell
        # The Circle is still accessible via grid.olds[0, 0]
        grid.mobs[0, 0] = Square(side_length=0.5, color=RED)

        # Transform the circle into the square.
        self.play(ReplacementTransform(grid.olds[0, 0], grid.mobs[0, 0]))
```  

---  

## More Examples  

See the following examples in the `/examples` directory:
1. [Getting Started](examples/01-getting_started.py)
2. [Buffers and Margins](examples/02-buffers_and_margins.py)
3. [Labels](examples/03-labels.py)
4. [Scrolling](examples/04-scrolling.py)
5. [Tagging](examples/05-tagging.py)
6. [Masking](examples/06-masking.py)
7. [Frame](examples/07-frame.py)

---  

## Internals  

To understand how to best interact with the Grid and why things go wrong sometimes, it is necessary to know more about its building blocks. This is only a general overview; for a deep dive, see the in-code documentation and the code itself.

### Cell  
The **Grid** class creates a two-dimensional layout of **Cell** objects.  
Each `Cell` holds:  
- `rect`: a `Rectangle` that defines the visual bounds of the cell.  
- `mob`: the current `Mobject` contained in the cell (defaults to an `EmptyMobject` placeholder).  
- `old`: the previous `Mobject` that occupied the cell, useful for transition animations.  
- `tags`: a dictionary-like `Tags` instance for arbitrary user-defined metadata.  

### Proxies  
The Grid class provides four proxy objects that give convenient, NumPy-style access to the underlying cell attributes described above. These proxies return as outputs and take in as inputs different types of objects whether you are targeting individual cells or multiple cells at the same time. e.g.: `grid.mobs[0,0]` returns a Mobject, while `grid.mobs[:]` returns a VGroup of all mobjects contained in the grid, in row-major order. The table below summarizes the expected inputs and the returned outputs for each proxy (single cell/bulk):

| Proxy        | Purpose                                    | Readable (`__getitem__`)     | Writeable (`__setitem__`)           |
|--------------|--------------------------------------------|------------------------------|-------------------------------------|
| `grid.mobs`  | Access or assign Mobject(s) to cell(s).    | ✅ Output: Mobject/VGroup    | ✅ Input: Mobject/Sequence[Mobject] |
| `grid.olds`  | Retrieve the previously stored Mobject(s). | ✅ Output: Mobject/VGroup    | ❌                                  |
| `grid.rects` | Access the lattice Rectangles.             | ✅ Output: Rectangle/VGroup  | ❌                                  |
| `grid.tags`  | Manipulate metadata via the `Tags` class.  | ✅ Output: STS/BTS objects*  | ✅ Input: Tags or mapping           |

*<sub>STS/BTS: ScalarTagsSelection/BulkTagsSelection objects returned when indexing the tags proxy (i.e. `grid.tags[...]`).</sub>  

### Tags  
Each `Cell` holds a `Tags` instance in their `tags` attribute. This class acts as a python dictionary with dot attribute access.
Moreover, the `ScalarTagsSelection` or `BulkTagSelection` instance returned by `TagsProxy.__getitem__` define the following methods (directly or through their base class `_TagsSelectionBase`):

| Method        | Purpose                                                          | Example                            | 
|---------------|------------------------------------------------------------------|------------------------------------|
| `update`      | Updates the `Tags` instance(s) similar to `dict.update`          | `grid.tags[...].update(foo="bar")` |
| `remove`      | Removes the provided keys from the `Tags` instance(s)            | `grid.tags[...].remove("foo")`     |
| `clear`       | Removes all keys from the `Tags` instance(s)                     | `grid.tags[...].clear()`           |
| `__getattr__` | Enables requesting the value(s) associated with the provided key | `grid.tags[...].foo`               |
| `__setattr__` | Enables setting the value(s) associated with the key             | `grid.tags[...].foo = "bar"`       |
| `__delattr__` | Enables deleting the provided key from the `Tags` instance(s)    | `del grid.tags[...].foo`           |

`ScalarTagsSelection.__setitem__` and `BulkTagSelection.__setitem__` on the other hand allow to completely replace the `Tags` instance(s). It accepts a new `Tags` instance (a copy will be set for each `Cell` in the bulk case) or a mapping (that will be internally converted into a `Tags` instance): `grid.tags[...] = Tags(foo="bar", baz=42)` and `grid.tags[...] = {"foo": "bar", "baz": 42}` or equivalent.

### Stencil Viewport and Scrolling  
When a `Grid` is instantiated with `num_visible_rows` and/or `num_visible_cols`, a [Stencil](https://github.com/maejam/manim-utils) instance is added to the grid and is stored in the `grid._stencil` attribute accesible via the `grid.stencil` property. Its purpose is to hide cells that should not be visible. It is a `manim.Difference` object between the whole grid and a `SurroundingRectangle` around the visible cells. The stencil is painted the same color as the scene background to give the impression that only the viewport is added to the screen. The stencil is always transformed with the grid and stays in sync with it.

The scrolling animation is actually the whole grid moving the opposite direction (scrolling DOWN is shifting the grid UP), while the viewport stays in place and the `Difference` is recomputed every frame with an updater. This is why scrolling past the last row/column gives weird artifacts: this is the result of the `Difference` between a `Mobject` and another one that does not entirely intersect it.

The following snippet shows how the `stencil` (YELLOW) covers all the hidden cells, while the `viewport` (RED) acts like a window on the visible cells. After scrolling DOWN (grid2 on the right), the whole grid is shifted UP along with the stencil, while the viewport stays in place:

```python
from manim import *

from manim_grid import Grid


class Stencil(Scene):
    def construct(self):
        grid = Grid(
            row_heights=[0.5] * 10,
            col_widths=[2] * 2,
            num_visible_rows=5,
        )
        self.add(grid.shift(LEFT))
        grid.mobs[:] = [Text(str(n), font_size=16) for n in range(20)]
        grid.add(grid.mobs[:])
        grid.lattice.set_stroke(opacity=1)
        grid.stencil.set_fill(YELLOW)
        grid.viewport.set_stroke(RED, opacity=1)
        grid2 = grid.copy()
        self.add(grid2.next_to(grid).scroll(DOWN, 3))
```

[stencil](assets/stencil.png)
