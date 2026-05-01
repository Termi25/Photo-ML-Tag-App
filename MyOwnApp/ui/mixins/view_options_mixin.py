"""ViewOptionsMixin — collapsible view-options bar (zoom, columns, file sort).

UI layout (collapsed by default)
─────────────────────────────────
[▶ View Options]  <zoom indicator>
  ─────────────────────────────────────────── (hidden until expanded)
  Zoom: ────────────────● px  [Auto]   Columns: [5 ↑↓]   Sort files: [name ▼]

The zoom slider (64–320 px) overrides the auto-sizing of gallery thumbnails.
The columns spinner (2–12) live-reflows the gallery without a refresh.
The file-sort dropdown controls the ordering in directory-browse mode
(independent of the search sort which is on the search bar itself).

State attributes used across the app
──────────────────────────────────────
self._view_zoom_override : int    0 = auto, >0 = fixed pixel size
self._view_gallery_cols  : int    current column count (default _GALLERY_COLS)
self._file_sort_var      : StringVar  current sort key for folder browsing

Methods owned by this mixin
───────────────────────────
_build_view_options_bar()   Construct the panel and initialise state attrs.
_toggle_view_options()      Collapse / expand content row.
_on_zoom_change(val)        Slider callback → update gallery if in icon mode.
_zoom_auto()                Reset zoom to auto.
_on_cols_change()           Spinner callback → reflow gallery.

Current location: file_explorer.py, class FileExplorer.
"""


class ViewOptionsMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer."""
