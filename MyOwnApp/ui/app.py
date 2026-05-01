"""ui/app.py — composed FileExplorer application class.

This module re-exports FileExplorer from the monolithic file_explorer.py so
that the rest of the codebase can import it from the clean package path:

    from ui.app import FileExplorer          # new path
    from file_explorer import FileExplorer   # legacy path (still works)

Migration plan
──────────────
As methods are moved from file_explorer.py into their respective mixin files
(ui/mixins/), this module will be updated to import the mixin versions.
Once all methods are migrated, FileExplorer will be declared here explicitly:

    from ui.mixins import (ThemeMixin, NavigationMixin, ViewsMixin, MLMixin,
                           SidebarMixin, SearchMixin, ViewOptionsMixin, DnDMixin)
    from ui.dialogs import TrainingProgressDialog, MoveProgressDialog, InboxTrainingDialog

    class FileExplorer(ThemeMixin, NavigationMixin, ViewsMixin, MLMixin,
                       SidebarMixin, SearchMixin, ViewOptionsMixin, DnDMixin):
        def __init__(self, root): ...
        def setup_ui(self): ...

Until that point, this shim keeps both import paths valid.
"""

from file_explorer import FileExplorer  # noqa: F401

__all__ = ['FileExplorer']
