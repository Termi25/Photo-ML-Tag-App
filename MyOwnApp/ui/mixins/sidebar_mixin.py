"""SidebarMixin — right-hand tag sidebar: display, add, and remove tags.

Methods owned by this mixin
───────────────────────────
setup_tag_sidebar(parent)       Build the tag-management panel.
refresh_tag_list()              Repopulate the tag list for selected image.
on_item_select(event)           Treeview selection → show image tags.
on_tree_select(event)           Alias / secondary selection handler.
add_manual_tag()                Read the "Add tag" entry and write to DB.
remove_tag_from_image()         Remove selected tag from DB + file metadata.
apply_folder_tags()             Apply hierarchy-derived tags to selected image.
show_image_tags(image_path)     Load and display DB tags for a path.

Current location: file_explorer.py, class FileExplorer.
"""


class SidebarMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer."""
