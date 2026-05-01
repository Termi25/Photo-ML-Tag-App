"""SearchMixin — MangaDex-style tag filter bar.

UX model
────────
1. Type a tag name → autocomplete popup (exact-start matches ranked first).
2. Select or press Enter → green chip (＋ TAG, included).
3. Click chip label → toggles to red (－ TAG, excluded).
4. Click chip × → removes chip.
5. Sort dropdown + ↑/↓ direction button on the same row.
6. Search re-executes automatically when chips change.
7. Uses ``data_layer.MetadataStore.search_images_advanced`` which supports
   both included AND excluded tags in a single efficient SQL query.

Methods owned by this mixin
───────────────────────────
_build_search_bar()             Build the two-row search panel.
_toggle_sort_dir()              Flip sort direction and re-run search.
_commit_search_token()          Add entry text as an included chip.
_add_tag_chip(tag, state)       Add / update a chip ('+' or '-').
_cycle_chip(tag)                Toggle include ↔ exclude; remove on third click.
_rebuild_chips()                Redraw the chip row from _tag_states.
_remove_chip(tag)               Remove a chip and optionally clear search.
_on_search_entry_change(*_)     Trace callback → trigger autocomplete.
_show_tag_suggestions(query)    Show / update the autocomplete popup.
_hide_tag_suggestions()         Withdraw the autocomplete popup.
_on_suggestion_select(event)    Apply selected suggestion as an included chip.
perform_tag_search()            Execute search_images_advanced + display results.
_display_search_results(results)
clear_search()                  Clear all chips and reload directory.

Current location: file_explorer.py, class FileExplorer.
"""


class SearchMixin:
    """Mixin stub — actual methods live in file_explorer.FileExplorer."""
