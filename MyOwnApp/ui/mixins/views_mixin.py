"""ViewsMixin — detailed list view and gallery (icon) view rendering.

Methods owned by this mixin
───────────────────────────
create_detailed_view()          Build the Treeview widget.
load_detailed_view(folders, files)
create_icons_view()             Build the scrollable canvas for gallery mode.
load_icons_view(folders, files)
_gallery_thumb_size()           Compute thumbnail pixel size from canvas width
                                and the View Options zoom/column settings.
_relayout_gallery()             Re-render gallery on canvas resize.
_create_icon_widget_fast(item, thumb_size)
_update_thumbnail_async(...)    Background thread: load and set real thumbnails.
get_file_type_from_extension(path)
create_thumbnail_for_tree(path, size)
_create_folder_collage(folder_path, thumb_size)
_collect_folder_preview_images(folder_path, max_images)
_zoom_fill(img, size, zoom)
_update_folder_preview(folder_path)  Regenerate and cache the folder collage PNG.

Current location: file_explorer.py, class FileExplorer.
"""


class ViewsMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer."""
