"""DnDMixin — drag-and-drop (OS and internal) plus clipboard cut/copy/paste.

OS drag-and-drop (tkinterdnd2)
──────────────────────────────
_setup_os_drop_target()     Register the content area as a DnD drop target.
_on_os_drop(event)          Handle files dropped from the OS file manager.

Internal drag-and-drop (mouse events on gallery / treeview items)
──────────────────────────────────────────────────────────────────
_start_drag(event)          Record drag source on Button-1 press.
_on_drag_motion(event)      Show ghost highlight while dragging.
_on_drag_release(event)     Execute move/copy when button is released.
_highlight_drop_target(iid) Visually mark a folder as the active drop zone.
_clear_drag_highlight()     Remove the drop zone marker.
_move_files_dnd(sources, target_dir)  Perform the actual file moves.

Clipboard
──────────
_copy_files(paths)          Mark paths for copy operation.
_cut_files(paths)           Mark paths for cut (move) operation.
_paste_files(target_dir)    Execute pending copy / cut into target_dir.

Context menu
─────────────
_show_context_menu(event)   Right-click menu for items in either view.
_on_right_click(event)      Alias used in icon view right-click binding.

Current location: file_explorer.py, class FileExplorer.
"""


class DnDMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer."""
