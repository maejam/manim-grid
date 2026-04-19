## v0.4.0 (2026-04-19)

### Feat

- **signals**: add `row_insertion_processed` and `row_insertion_displayed` signals
- **signals**: add `column_insertion_processed` and `column_insertion_displayed` signals
- **viewport**: add predicate to `_compute_viewport` and `update_viewport`
- **BaseProxy**: define `__len__` on proxies
- **numbers**: `row_numbers` and `col_numbers` accept `start`/`stop`/`step`
- **insert**: add `insert_row` and `insert_column` methods
- **align**: alignment defaults to the last alignment in each cell
- **MobsProxy**: allow VGroup as a value in `MobsProxy.__setitem__`
- **scroll**: add `free_scroll` method
- **signals**: add `tag_changed` signal
- **signals**: add `mobs_added` and `mob_inserted` signals
- **signals**: add signals system
- **Grid**: add `Grid.fullscreen` alternative constructor

### Fix

- **frame**: include viewport stroke width when updating the frame and calculating margins
- **viewport**: include inner rects and viewport stroke width when updating viewport
- **stencil**: make the stencil correctly wrap the grid by including the `viewport` in `Stencil._wrapped`
- **viewport**: remove `viewport` as a `Grid` submobject: `keep_viewport_static` fixed
- **labels**: accessing `row_labels` and `col_labels` raise `GridLabelsError` when no labels are defined
- **frame**: define the frame as a `Grid` submobject, not a `viewport` one

### Refactor

- **viewport**: `Grid.viewport` is now the stencil clip instead of a copy
- **labels**: dissociate row/col labels and row/col numbers
- **TagsProxy**: make `TagsProxy` read-only
- **TagsProxy**: replace `Selection` classes with `Tags` and `TagsList`

## v0.3.0 (2026-03-27)

### Feat

- **frame**: add frame mobject around viewport
- **labels**: add `Grid.row_labels` and `Grid.col_labels`
- **RectsProxy**: add `RectsProxy` to expose `rect` attribute
- **scroll**: add viewport and scrolling

### Fix

- **typing**: `ListKey` type alias
- **stencil**: the stencil no longer covers mobjects added after the `Grid`
- **MobsProxy**: `MobsProxy` index could be invalid and not pass assertion

### Refactor

- **rects**: rename the VGroup containing the Rectangles (frame->lattice)
- **Grid**: rely on opacity tricks rather than `_all` group

## v0.2.0 (2025-12-16)

### Feat

- **TagsProxy**: add `TagsProxy` to expose `tags` attribute
- **BaseProxy**: make proxies iterable
- **BaseProxy**: add masking functionality
- **BaseProxy**: add proxies to expose `mobs` and `olds` `Cell` attributes
- **Grid**: add `Cell` class and `Grid` façade
- **labels**: add `LabelMapper` to translate string labels to indices
