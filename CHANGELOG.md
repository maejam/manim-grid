## 0.3.0 (2026-03-27)

### Feat

- **frame**: add frame mobject around viewport
- **labels**: add `Grid.row_labels` and `Grid.col_labels`
- **rectsproxy**: add RectsProxy
- **scroll**: add viewport and scrolling

### Fix

- **scroll**: scrolling without `animate` did not update the stencil
- **typing**: ListKey type alias
- **stencil**: the Stencil no longer covers mobjects added after the Grid
- **mobsproxy**: MobsProxy index could be invalid and not pass assertion

### Refactor

- **stencil**: stencil and viewport management
- **rects**: rename the VGroup containing the Rectangles (frame->lattice)
- **grid**: rely on opacity tricks rather than _all group

## 0.2.0 (2025-12-16)

### Feat

- **tagsproxy**: add TagsProxy
- **proxies**: make proxies iterable
- **proxies**: add masking functionality
- **grid**: add Cell class and Grid façade
- **proxies**: add proxies to expose Cell attributes.
- **labels**: add LabelMapper
