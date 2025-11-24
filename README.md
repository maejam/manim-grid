# 📐 `manim-grid` - A Simple Grid Container to ease Mobjects positioning  

`manim-grid` is a lightweight [Manim](https://www.manim.community/) plugin that creates a rectangular grid of invisible cells. It lets you place any `Mobject` into a cell by using natural indexing (`grid[row, col]`), and handles alignment, margins, and automatic positioning for you.

> **Why this plugin?**  
> It is born from an attempt to build a better `Code` Mobject, more flexible and easier to work with. Manim’s built‑in `arrange_in_grid` arranges submojects but it doesn’t give you a persistent “cell” abstraction you can address later. `Grid` fills that gap: each cell is a `Rectangle` with zero‑opacity that you can treat as a slot for other objects, retrieve later, and re‑assign on the fly.

---

## Table of Contents  

- [Features](#features)  
- [Installation](#installation)  
- [Quick Start](#quick start)  
- [API Reference](#api reference)  
  - [`Grid.__init__`](#gridinit)  
  - [`Grid.__getitem__`](#gridgetitem)  
  - [`Grid.__setitem__`](#gridsetitem)  
- [More to Come](#more to come)  

---

## Features  

- **Declarative geometry** – specify per-row heights and per-column widths.  
- **Automatic layout** – cells are arranged with `arrange_in_grid`.  
- **Margin & buffer control** – fine‑tune spacing between cells and padding inside cells.  
- **Pythonic indexing** – `grid[row, col]` returns the stored `Mobject`; `grid[row, col] = obj` places an object. Negative indices are allowed and behave like normal Python sequences.  
- **Alignment vectors** – optionally align a `Mobject` to any edge of a cell (`grid[row, col, UP]`, `grid[row, col, DL]`, etc.).  

---

## Installation  

For now there is no Pypi package. Install by adding to your `manim` project:
- create the project if necessary:
```bash
uv init myproject
cd myproject
```
- add the plugin to your newly created or existing project with all the available services:
```bash
uv add git+https://github.com/maejam/manim-grid.git
```

## Quick Start  

```python
from manim import *
from manim_grid import Grid

class GridDemo(Scene):
    def construct(self):
        # 2 rows × 3 columns with different sizes
        grid = Grid(
            row_heights=[1.0, 1.5],
            col_widths=[1.5, 1, 1.5],
            buff=0.2,  # space between cells - pass a 2-tuple to dissociate horizontal/vertical buffers
            margin=0.15,  # inner padding inside each cell - pass a 2-tuple to dissociate horizontal/vertical margins
        )
        self.add(grid)

        # Place a circle in the top-right cell, centred
        circle = Circle(radius=0.3, color=BLUE)
        grid[0, 2] = (
            circle  # The Mobject is deliberatly not added to the scene (nor is the previous occupant of that cell, if any, removed) to allow for greater control over animations
        )
        self.add(circle)

        # Place a square aligned to the lower-left corner of the bottom-left cell
        square = Square(side_length=0.2, color=RED)
        grid[1, 0, DL] = square
        self.add(square)

        # Retrieve later
        c = grid[0, 2]
        assert c is circle
        self.wait()

        # Show the grid
        grid.grid.set_stroke(opacity=0.5)
        self.wait()
```

---

## API Reference  

### `Grid.__init__(self, row_heights, col_widths, buff=0.0, margin=0.1, **kwargs)`  

| Parameter     | Type                            | Description                                                            |
|---------------|---------------------------------|------------------------------------------------------------------------|
| `row_heights` | `Sequence[float]`               | Height of each row (top-to-bottom).                                    |
| `col_widths`  | `Sequence[float]`               | Width of each column (left-to-right).                                  |
| `buff`        | `float` \| `tuple[float,float]` | Gap between cells (a single float is used for both directions).        |
| `margin`      | `float` \| `tuple[float,float]` | Padding inside each cell (a single float is used for both directions). |
| `**kwargs`    | –                               | Passed straight to the parent `Mobject`                                |

Attributes:  
- `self.cells` – a list-of-lists of `Rectangle`s. Gives access to individual cells: ``grid.cells[row_num][col_num]``  
- `self.grid` – a flat `VGroup` containing the same rectangles, arranged in a grid.  

---

### `Grid.__getitem__(self, idx) -> m.Mobject`  

    obj = grid[row, col]

Returns the `Mobject` stored at the given cell.  
* Supports negative indices (`-1` -> last row/column).  
* Raises `IndexError` if the indices are out of bounds.  

---

### `Grid.__setitem__(self, idx, mob) -> None`  

    grid[row, col] = mob                     # centre alignment (default)
    grid[row, col, alignment] = mob          # custom alignment vector
    grid[row, col] = Mobject()               # clear the cell content

* `idx` can be a 2-tuple `(row, col)` or a 3-tuple `(row, col, alignment)`.  
* `alignment` is a `Vector3D` (e.g. `UP`, `DR`, ...).  
* Performs the same bounds checking as `__getitem__`.  
* `Grid` is only a positioning helper. Setting a mobject in a cell does not add it to the Scene and does not remove the previous mobject in that cell from the Scene. This is intentional to give more control to the user (e.g. using `Transform` to morph the old mobject into the new one).

---

## More to come

- Grids that span the whole screen.
- Specialized Grids for text and code.
- Indexing rows and columns with labels.

Contributions and Ideas are welcomed!
