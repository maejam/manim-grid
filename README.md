# 📐 `manim-grid` - A Simple Grid Container to ease Mobjects positioning and referencing  

`manim-grid` is a lightweight [Manim](https://www.manim.community/) plugin that creates a rectangular grid of cells. It lets you place any `Mobject` into a cell by using natural indexing and handles alignment, margins, and automatic positioning for you. As it is based on [NumPy](https://numpy.org/), the full power of NumPy indexing is available.  

> **Why this plugin?**  
> It is born from an attempt to build a better `Code` Mobject, more flexible and easier to work with. Manim’s built‑in `arrange_in_grid` arranges submojects but it doesn’t give you a persistent “cell” abstraction you can address later. `Grid` fills that gap: each cell is a slot in which you can store other mobjects, retrieve them later, and re-assign on the fly.  
---  

## Table of Contents  

- [Features](#features)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
- [Core Concepts](#core-concepts)  
  - [Grid & Cell](#grid-&-cell)  
  - [Proxies (`mobs`, `olds`, `tags`)](#proxies)  
  - [Masking](#masking)  
- [More to come](#more-to-come)  

---  

## Features  

- **Declarative geometry** - specify per-row heights and per-column widths.  
- **Automatic layout** - cells are arranged with `arrange_in_grid`.  
- **Margin & buffer control** - fine-tune spacing between cells and padding inside cells.  
- **Pythonic NumPy-based indexing** - `grid.mobs[...]` returns the stored `Mobject`(s); `grid.mobs[...] = obj` places mobject(s). Negative indices, slices, masks are allowed and behave like normal NumPy arrays.  
- **String labels** - string identifiers can be added to rows and columns to make indexing more expressive.  
- **Access previous Mobjects** - `grid.olds[...]` returns the previous mobject(s) in a cell or a group of cells. Useful for animations.  
- **Alignment vectors** - align a `Mobject` to any edge/corner in a cell (`grid[row, col, UP]`, `grid[row, :, DL]`, etc.).  
- **Per-cell metadata** - add key/value tags to cells, individually or in bulk.  
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

## Quick Start  

```python
from manim import *
from manim_grid import Grid


class QuickStart(Scene):
    def construct(self):
        # Create a 2×3 grid (rows, columns)
        grid = Grid(
            row_heights=[2, 2],
            col_widths=[2, 2, 2],
            row_labels=["top", "bottom"],
            col_labels=["left", "mid", "right"],
        )

        # Place mobjects in the top row, aligned to the upper edge.
        # The Mobjects are deliberatly not added to the scene (nor are the previous
        # occupants of that cells, if any, removed) to allow for greater control
        # over animations.
        grid.mobs["top", :, UP] = [
            Circle(radius=0.5, color=BLUE),
            Dot(color=GREEN),
            Rectangle(height=0.3, width=0.5),
        ]
        self.add(grid.mobs["top"])

        # Place a square in the top-left cell, centered (default).
        grid.mobs["top", "left"] = Square(side_length=0.5, color=RED)

        # Transform the circle into the square.
        self.play(
            ReplacementTransform(grid.olds["top", "left"], grid.mobs["top", "left"])
        )

        # Tag the whole top row for later reference.
        grid.tags["top"].to_remove = True

        # Show the grid.
        self.add(grid)
        grid.grid.set_stroke(opacity=0.5)
        self.wait()

        # Remove all mobjects in the tagged cells using a mask.
        # The mobjects are still in the cells but removed from the scene.
        mask = grid.tags.mask(to_remove=True)
        self.remove(*grid.mobs[mask])
        self.wait()
```

---  

## Core concepts  

### Grid & Cell  
The **Grid** class creates a two-dimensional layout of **Cell** objects.  
Each `Cell` holds:  
- `rect`: a `manim.Rectangle` that defines the visual bounds of the cell.  
- `mob`: the current `manim.Mobject` displayed in the cell (defaults to an `EmptyMobject` placeholder).  
- `old`: the previous `Mobject` that occupied the cell, useful for transition animations.  
- `tags`: a dictionary-like `Tags` instance for arbitrary user-defined metadata.  

The grid takes sequences of row heights and column widths, optional buffers (between cells), and optional margins (inside cells).  

```python
from manim_grid import Grid

# Create a 2x3 grid with uniform cell sizes.
grid = Grid(
    row_heights=[2, 2],
    col_widths=[3, 3, 3],
    buff=0.2,
    margin=0.1,
)
```
- `buff` and `margin` can be scalar values, in which case the same value is used for the horizontal and vertical dimensions. Alternatively, a 2-tuple `(horizontal, vertical)` can be specified.  
- `row_labels` and `col_labels` sequences can be passed as well (see `Quick Start`). If so, they must be the same length as `row_heights` and `col_widths` respectively. If omitted, 1-based string numeric labels will be automatically created (e.g. `grid.mobs[0, 0]` is equivalent to `grid.mobs["1", "1"]`).  

### Proxies  

Manim‑Grid provides three proxy objects that give convenient, NumPy-style access to the underlying cell attributes.  

| Proxy       | Purpose                                    | Readable (`__getitem__`)     | Writeable (`__setitem__`)           |
|-------------|--------------------------------------------|------------------------------|-------------------------------------|
| `grid.mobs` | Access or assign Mobject(s) to cell(s).    | ✅ Output: Mobject/VGroup    | ✅ Input: Mobject/Sequence[Mobject] |
| `grid.olds` | Retrieve the previously stored Mobject(s). | ✅ Output: Mobject/VGroup    | ❌                                  |
| `grid.tags` | Manipulate metadata via the `Tags` class.  | ✅ Output: STS/BTS objects*  | ✅ Input: Any or mapping            |

*<sub>STS/BTS: ScalarTagsSelection/BulkTagsSelection objects returned when indexing the tags proxy (i.e. `grid.tags[...]`).</sub>  
The `Output` and `Input` types correspond to `single cell/bulk` except for `tags.__setitem__` (see examples below).  


#### Example: Adding and retrieving Mobjects  

```python
from manim import Circle, UP

# Place a Circle in row 0, column 1, aligned to the top edge of the cell.
grid.mobs[0, 1, UP] = Circle(radius=0.5)

# Bulk assign Mobjects to the first column, centered.
grid.mobs[:, 0] = [Dot(color=RED), Dot(color=BLUE)]

# Retrieve all Mobjects in the grid. Returns a VGroup where Mobjects are ordered row-major.
all_mobs = grid.mobs[:]
print(list(all_mobs)) # [Dot, Circle, EmptyMobject, Dot, EmptyMobject, EmptyMobject]
```  

#### Example: Reading old objects  
```python

# olds are read-only and automatically managed by the library.
grid.mobs[0, 1] = Square() # Replaces the Circle which becomes the `old` mobject in that cell.
old_mobs = grid.olds[:]
print(list(old_mobs)) # [EmptyMobject, Circle, EmptyMobject, EmptyMobject, EmptyMobject, EmptyMobject]

```  

#### Example: Tagging cells  
```python

# Set a custom tag on cell(s). Here a single cell.
grid.tags[0, 1].foo = "bar"

# Bulk-update cell(s). Here a whole row.
grid.tags[0].update(baz=True, qux=42)

# Retrieve a tag array for cell(s). Here the entire grid.
baz_tags = grid.tags[:].baz   # Returns a NumPy ndarray with the `baz` value for each cell
print(baz_tags) # [[True True True] [<MISSING> <MISSING> <MISSING>]]

# Removing a single tag on cell(s).
grid.tags[0, 0].remove("baz")

# Clearing all the tags from cell(s).
grid.tags[0, 1].clear()

# Replacing all the tags on cell(s). Accepts a `Tags` instance or a Mapping.
grid.tags[1] = {"baz": False, "foo": 24}
```

### Masking  

All proxies support a .mask() method that builds a Boolean mask based on a predicate or attribute filters. Both predicate and filters must be True for a cell to be selected. If a cell is missing a filter attribute, it will not be selected.  

```python

# Select all cells whose current mob is a Circle.
circle_mask = grid.mobs.mask(predicate=lambda mob: isinstance(mob, Circle))

# Apply a tag only to those cells.
grid.tags[circle_mask].update(shape="circle")

# Select all cells where the `old` mob is a RED Square with opacity 1.
red_mask = grid.olds.mask(predicate=lambda old: isinstance(old, Square), color=RED, opacity=1)

# Remove all the selected old mobs from the scene.
self.remove(*grid.olds[red_mask])
```

---  

## More to come  

- Specialized Grids for text and code.  

Feedback, contributions and ideas are welcomed!  
