"""ui.mixins — method-group mixins for FileExplorer.

Each mixin file owns one functional slice of the FileExplorer class.
FileExplorer (defined in ui/app.py) inherits from all of them via MRO.

Mixin              Responsibility
─────────────────  ────────────────────────────────────────────────────────────
ThemeMixin         _apply_theme, change_theme, theme colour constants
NavigationMixin    populate_drives, go_back/up/home, load_directory, refresh
ViewsMixin         create/load detailed & icon views, thumbnail generation
MLMixin            ML control panel, training/sorting/clustering actions
SidebarMixin       Tag sidebar: display, add/remove tags
SearchMixin        MangaDex-style tag filter bar (include / exclude chips)
ViewOptionsMixin   View options bar: zoom slider, column count, file sort
DnDMixin           OS drop target, internal drag-and-drop, clipboard cut/copy/paste

NOTE: the mixin source files currently re-export symbols from the monolithic
      file_explorer.py.  They exist as stable import targets; methods will be
      migrated into them incrementally without breaking callers.
"""

# Re-export the mixin classes so ``from ui.mixins import SearchMixin`` works
# regardless of which stage of the migration we are in.
from .theme_mixin        import ThemeMixin
from .navigation_mixin   import NavigationMixin
from .views_mixin        import ViewsMixin
from .ml_mixin           import MLMixin
from .sidebar_mixin      import SidebarMixin
from .search_mixin       import SearchMixin
from .view_options_mixin import ViewOptionsMixin
from .dnd_mixin          import DnDMixin

__all__ = [
    'ThemeMixin', 'NavigationMixin', 'ViewsMixin', 'MLMixin',
    'SidebarMixin', 'SearchMixin', 'ViewOptionsMixin', 'DnDMixin',
]
