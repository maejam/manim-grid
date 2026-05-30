> [!WARNING]
> While the library is in major version 0, the API is not considered stable and can change without notice.  


# 📐 `manim-grid` - A powerful Grid Container to ease Mobjects positioning and referencing  

`manim-grid` is a [Manim](https://www.manim.community/) plugin that offers a new way to build scenes and interact with manim. It creates a rectangular grid of cells in which you can place any `Mobject` by using a powerful, [NumPy](https://numpy.org/)-based natural indexing. It handles mobject positioning, automation and provides custom animations.  

It is born from an attempt to build a better `Code` Mobject, more flexible and easier to work with. This goal quickly turned into a flexible multi-purpose tool. It can be used as a ruler to position mobjects in a scene, retrieve them later, and re-assign them on the fly. It can also be used to build event-based scenes.  

---  

## Table of Contents  

- [Features](#features)  
- [Installation](#installation)  
- [Getting Started](#getting-started)  
- [More Examples](#more-examples)
- [Tips](#tips)
- [Internals](#internals)  
  - [Cell](#cell)  
  - [Proxies](#proxies)  
  - [Tags/Config](#tags-/-config)
  - [Stencil Viewport and Scrolling](#stencil-viewport-and-scrolling)
  - [Signals](#signals)

---  

## Features  

- **Declarative geometry** - specify per-row heights and per-column widths.  
- **Automatic layout** - cells are arranged with `arrange_in_grid`.  
- **Margin & buffer control** - fine-tune spacing between cells and padding inside cells.  
- **Pythonic NumPy-based indexing** - the full power of NumPy indexing is supported: negative indices, slices, masks...
- **String labels** - string identifiers can be added to rows and columns to make indexing even more expressive.  
- **Alignment vectors** - align `Mobjects` to any edge/corner in cells.  
- **Powerful management of cell attributes** - access any attribute for a single cell or in bulk as you would a numpy array.  
- **Per-cell metadata** - add key/value tags to cells.  
- **Scrolling** - the `scroll` method lets you scroll the grid in any direction with a smooth animation.  
- **Inserting** - the `insert_row` and `insert_column` methods let you..., well..., insert rows and columns. With full control over the visual result.  
- **Frame** - add a custom frame around the grid that plays well with its dynamic nature.  
- **Signals** - react to events to automate the Grid behaviour.  
- **Per cell configuration** - each cell can be configured and updated independently.  
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
> It is necessary to be familiar with [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html) to fully benefit from this plugin.  

https://github.com/user-attachments/assets/f1364a89-95dc-4109-bbc1-510232f87ecc  

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
        # mobjects in those cells, if any, removed) to allow for greater control
        # over animations.
        grid.mobs[0, :, UP] = [
            Circle(radius=0.5, color=BLUE),
            Dot(color=GREEN),
            Rectangle(height=0.3, width=0.5),
        ]
        grid.add(*grid.mobs[0])

        # Place a square in the top-left cell, centered
        # It replaces the Circle in that cell
        # The Circle is still accessible via grid.olds[0, 0]
        grid.mobs[0, 0, ORIGIN] = Square(side_length=0.5, color=RED)

        # Transform the circle into the square.
        self.play(ReplacementTransform(grid.olds[0, 0], grid.mobs[0, 0]))
```  

---  

## More Examples  

The following examples can be found in the [examples](examples/) directory. They are meant to serve as documentation/tutorials on each topic:  
 - [Getting Started](examples/getting_started.py)
 - [Buffers and Margins](examples/buffers_and_margins.py)
 - [Labels](examples/labels.py)
 - [Scrolling](examples/scrolling.py)
 - [Inserting rows/columns](examples/inserting.py)
 - [Tagging](examples/tagging.py)
 - [Masking](examples/masking.py)
 - [Frame](examples/frame.py)
 - [Alternative Constructors](examples/alternative_constructors.py)
 - [Signals](examples/signals.py)
 - [Config and Update](examples/config_and_update.py)
 - [Builtin Handlers](examples/builtin_signal_handlers.py)

---  

### Tips  

- It is necessary to understand the distinction between the Grid internal state (a NumPy array of Cells) and the Grid submobjects. When attributing mobjects to cells (`grid.mobs[...] = ...`), **they are not added as submobjects** to the Grid, which means they will not be visible and will not be transformed with the Grid. A second step is necessary: `grid.add(grid.mobs[...])`. This design is intentional for greater control over transitions and animations. This second step (adding as submobjects) can be automated with Signals.
- It is **highly recommended to unpack any Group/VGroup when adding/removing submobjects**. i.e., `grid.add(*grid.mobs[...])` is prefered over `grid.add(grid.mobs[...])`. This is because `grid.mobs[...]` returns a VGroup when multiple mobjects are targeted. Adding it directly would require to keep a reference to that Group instance to remove it later. Unpacking it gives more control over removing individual mobjects later. Because this is easy to forget, a warning is logged when adding or removing Groups without unpacking them.  

---  

## Internals  

To understand how to best interact with the Grid and why things go wrong sometimes, it is necessary to know more about its building blocks. This is only a general overview; for a deep dive, see the in-code documentation and the code itself.

### Cell  
The **Grid** class creates a two-dimensional layout of **Cell** objects.  
Each `Cell` holds:  
- `mob`: the current `Mobject` contained in the cell (defaults to an `EmptyMobject` placeholder).  
- `old`: the previous `Mobject` that occupied the cell, useful for transition animations.  
- `rect`: a `Rectangle` that defines the visual bounds of the cell.  
- `tags`: a dictionary-like `Tags` instance for arbitrary user-defined metadata.  
- `config`: a dictionary-like `Config` instance holding cells config options.  
- `updater`: a `CellUpdater` instance that allows to apply the config options.  


### Proxies  
The Grid provides access to the underlying NumPy array of cell objects via `grid.cells`. It also defines proxy objects that give convenient, NumPy-style access to the cell attributes described above. These proxies return as outputs and take in as inputs different types of objects when targeting individual cells or multiple cells at the same time. e.g.: `grid.mobs[0,0]` returns a Mobject, while `grid.mobs[:]` returns a VGroup of all mobjects contained in the grid, in row-major order. The table below summarizes the expected inputs and the returned outputs for each proxy (single cell/bulk):

| Proxy         | Purpose                                  | Readable (`__getitem__`)     | Writeable (`__setitem__`)                 |
|---------------|------------------------------------------|------------------------------|-------------------------------------------|
|`.mobs`        |Access or assign Mobject(s) to cell(s).   |✅ Output: Mobject/VGroup     |✅ Input: Mobject/Sequence[Mobject]|(V)Goup|
|`.olds`        |Retrieve the previously stored Mobject(s).|✅ Output: Mobject/VGroup     |❌                                         |
|`.rects`       |Access the lattice Rectangles.            |✅ Output: Rectangle/VGroup   |❌                                         |
|`.tags`        |Store and manipulate metadata in Cells.   |✅ Output: Tags/TagsList      |❌                                         |
|`.config`      |Set the config options in cell(s).        |✅ Output: Config/ConfigList  |❌                                         |
|`.update_cells`|Force an update based on the config.      |✅ Output: CellUpdater/CellUpdaterList|❌                                 |


### Tags / Config  
Each `Cell` holds a `Tags` instance in their `tags` attribute and a `Config` instance in their `config` attribute. These 2 classes are subclasses of `manim_grid.proxies.map_list.Map` which is a `MutableMapping` with dot attribute access and key validation.
This `Tags`/`Config`  instance is returned when indexing a single `Cell`, whereas a `TagsList`/`ConfigList` instance is returned when indexing multiple cells (subclasses of `manim_grid.proxies.map_list.MapList`). `MapList` is also a `MutableMapping` but holds list of the scalar value type as values. `MapList` also allows to broadcast scalar values to lists of scalar values for convenince. This allows to interact with a `Map` and `MapList` instances with the same Mapping interface.  

For example, `grid.tags[:].update(foo="bar")` will update all `Tags` instances in the `TagsList` returned by the `TagsProxy` (the `grid.tags[:]` part). Similarly, `grid.tags[:].pop("foo")` will pop the `foo` key from all the child `Tags` in the `TagsList` and will return the popped values in a list. If the `foo` key is missing in any `Tags`, it will raise a `KeyError`. Pass a default value to avoid this: `grid.tags[:].pop("foo", MISSING)`.  

All dict methods work on `Mapist` in a similar way they work on `Map`.  

### Stencil Viewport and Scrolling  
When a `Grid` is instantiated with `num_visible_rows` and/or `num_visible_cols`, a [Stencil](https://github.com/maejam/manim-utils) instance is added to the grid and is stored in the `grid._stencil` attribute accesible via the `grid.stencil` property. Its purpose is to hide cells that should not be visible. It is a `manim.Difference` object between the whole grid and a `Rectangle` surrounding the visible cells that defines the viewport boundaries. The stencil is painted the same color as the scene background to give the impression that only the viewport is added to the screen. The stencil is always transformed with the grid and stays in sync with it. The `viewport`, i.e. the clip in the `Stencil` usually stays in sync as well, except when using two context managers:
 - `Grid.keep_viewport_static` allows the Grid to move around while the viewport stays in place. It is used for scrolling for example: the scrolling animation is actually the whole grid moving the opposite direction (scrolling DOWN is shifting the grid UP), while the viewport stays in place and the `Difference` is recomputed every frame with an updater.
 - `Grid.update_viewport` allows the viewport to be dynamically resized to still encompass `num_visible_rows` and `num_visible_cols` while rows/columns are being resized for example.  

The following snippet shows how the `stencil` (YELLOW) covers all the hidden cells, while the `viewport` (RED) acts like a window on the visible cells. After scrolling DOWN (grid2 on the right), the whole grid is shifted UP along with the stencil, while the viewport stays in place:  

<img width="854" height="480" alt="stencil" src="https://github.com/user-attachments/assets/37f1096b-970a-4167-82e6-c52990fab31e" />  

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

### Signals
The signals system is implemented with the great [blinker](https://blinker.readthedocs.io/en/stable/) library.
See the available signals and their parameters in the [signals.py](src/manim_grid/signals.py) module.
