"""ThemeMixin — XML-driven theme application and runtime switching.

Methods owned by this mixin
───────────────────────────
_apply_theme()          Load ThemeManager colours into ttk.Style and expose
                        self._ACCENT / _BG / _CARD etc. shortcuts.
change_theme(name)      Load a different XML theme and repaint the window.

All theme XML files live in ui/theme/themes/.  ThemeManager is instantiated
in FileExplorer.__init__ (ui/app.py) before setup_ui() is called.

Current location of implementation: file_explorer.py, class FileExplorer,
section "Theme (XML-driven via ui.theme.ThemeManager)".
"""


class ThemeMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer.

    Once the migration is complete, move _apply_theme and change_theme here
    and remove them from file_explorer.py.
    """
    # _apply_theme and change_theme are defined in file_explorer.FileExplorer
    # until the migration is complete.
