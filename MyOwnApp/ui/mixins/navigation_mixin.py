"""NavigationMixin — directory traversal, drive panel, toolbar actions.

Methods owned by this mixin
───────────────────────────
populate_drives()               Enumerate OS drives into the left panel.
on_drive_select(event)          Handle drive selection.
navigate_to_path()              Navigate to the path typed in the address bar.
go_back()                       Navigate to previous directory.
go_up()                         Navigate to parent directory.
go_home()                       Navigate to the root folder.
refresh()                       Reload current directory.
load_directory(path)            Main entry point for loading a folder.
switch_to_detailed()            Switch content area to detailed (list) view.
switch_to_icons()               Switch content area to gallery (icon) view.
show_detailed_view()            Show the detailed Treeview widget.
show_icons_view()               Show the gallery Canvas widget.
set_root_folder_dialog()        Open folder picker and set new root.
initialize_ml_system()          Init the ML controller for the selected root.
clean_orphan_databases_action() Menu action to remove stale DB files.
reset_folder_previews()         Delete all cached folder collage previews.

Current location: file_explorer.py, class FileExplorer.
"""


class NavigationMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer."""
