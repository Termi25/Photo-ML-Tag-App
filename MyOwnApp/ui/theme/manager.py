"""ThemeManager — load an XML theme file and apply it to a Tkinter root.

Usage
-----
    from ui.theme import ThemeManager

    mgr = ThemeManager()               # auto-discovers themes/ folder
    mgr.load('light')                  # load by name or full path
    mgr.apply(root)                    # apply to Tk root window

    # Switch theme at runtime
    mgr.load('dark')
    mgr.apply(root)

    # List available themes
    names = mgr.available_themes()     # ['light', 'dark', 'high_contrast']
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import ttk


# Built-in themes directory sits next to this file
_BUILTIN_THEMES_DIR = Path(__file__).parent / 'themes'


class ThemeManager:
    """Parses a theme XML and applies it to a ``ttk.Style`` instance.

    The XML schema is documented in ``themes/light.xml``.  Every ``<colors>``,
    ``<fonts>``, and ``<layout>`` element is parsed into dictionaries; the
    ``apply()`` method then configures all standard ttk widget styles in one
    pass.  Calling ``apply()`` again after ``load()`` swaps the running theme
    without restarting the application.
    """

    def __init__(self, themes_dir: Optional[Path] = None) -> None:
        self._themes_dir: Path        = themes_dir or _BUILTIN_THEMES_DIR
        self._name:       str         = ''
        self._colors:     Dict[str, str]   = {}
        self._fonts:      Dict[str, tuple] = {}
        self._layout:     Dict[str, Any]   = {}
        self._base_ttk:   str         = 'clam'

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def available_themes(self) -> list[str]:
        """Return theme names (XML stem) found in the themes directory."""
        if not self._themes_dir.is_dir():
            return []
        return sorted(p.stem for p in self._themes_dir.glob('*.xml'))

    def theme_path(self, name: str) -> Optional[Path]:
        """Resolve a theme name to its XML path, or return None if not found."""
        candidate = self._themes_dir / f'{name}.xml'
        if candidate.exists():
            return candidate
        # Maybe the caller passed a full path
        p = Path(name)
        return p if p.exists() else None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, name_or_path: str) -> 'ThemeManager':
        """Parse *name_or_path* (theme name or absolute path to an XML file).

        Returns *self* for chaining: ``mgr.load('dark').apply(root)``.
        """
        path = self.theme_path(name_or_path)
        if path is None:
            raise FileNotFoundError(
                f"Theme '{name_or_path}' not found. "
                f"Available: {self.available_themes()}"
            )
        self._parse(path)
        return self

    def _parse(self, path: Path) -> None:
        tree = ET.parse(str(path))
        root = tree.getroot()

        meta = root.find('meta')
        self._base_ttk = (meta.findtext('base_ttk_theme') or 'clam') if meta else 'clam'
        self._name     = root.get('name', path.stem)

        # ── Colors ───────────────────────────────────────────────────────
        self._colors = {}
        colors_el = root.find('colors')
        if colors_el is not None:
            for child in colors_el:
                if child.text:
                    self._colors[child.tag] = child.text.strip()

        # ── Fonts ─────────────────────────────────────────────────────────
        self._fonts = {}
        fonts_el = root.find('fonts')
        if fonts_el is not None:
            for font_el in fonts_el.findall('font'):
                name   = font_el.get('name', 'default')
                family = font_el.get('family', 'Segoe UI')
                size   = int(font_el.get('size', '10'))
                weight = font_el.get('weight', '')
                self._fonts[name] = (family, size, weight) if weight else (family, size)

        # ── Layout ────────────────────────────────────────────────────────
        self._layout = {}
        layout_el = root.find('layout')
        if layout_el is not None:
            for child in layout_el:
                if child.text:
                    try:
                        self._layout[child.tag] = int(child.text.strip())
                    except ValueError:
                        self._layout[child.tag] = child.text.strip()

    # ------------------------------------------------------------------
    # Accessors (used by widgets that need raw values)
    # ------------------------------------------------------------------

    def color(self, name: str, fallback: str = '#888888') -> str:
        return self._colors.get(name, fallback)

    def font(self, name: str = 'default') -> tuple:
        return self._fonts.get(name, ('Segoe UI', 10))

    def layout(self, name: str, fallback: Any = 0) -> Any:
        return self._layout.get(name, fallback)

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    def apply(self, root: tk.Tk) -> None:
        """Configure *root* and all ttk widgets with the loaded theme.

        Safe to call multiple times — each call fully replaces the previous
        style configuration, making runtime theme switching possible.
        """
        if not self._colors:
            raise RuntimeError("No theme loaded. Call load() first.")

        c  = self._colors
        f  = self._fonts
        ly = self._layout

        style = ttk.Style(root)
        style.theme_use(self._base_ttk)

        bg   = c.get('background',   '#f3f3f3')
        card = c.get('card',         '#ffffff')
        fg   = c.get('foreground',   '#1a1a1a')
        fg_d = c.get('foreground_dim','#555555')
        acc  = c.get('accent',       '#0078d4')
        acc_h = c.get('accent_hover','#106ebe')
        acc_p = c.get('accent_pressed', '#005a9e')
        hov  = c.get('hover',        '#e5e5e5')
        pre  = c.get('pressed',      '#d0d0d0')
        brd  = c.get('border',       '#d0d0d0')
        sel_bg = c.get('selection_bg', acc)
        sel_fg = c.get('selection_fg', '#ffffff')
        trough = c.get('trough',     '#e0e0e0')
        prog   = c.get('progressbar', acc)
        th_bg  = c.get('treeview_heading_bg', '#e8e8e8')
        sb_bg  = c.get('status_bar_bg', '#e0e0e0')

        px = int(ly.get('button_padding_x', 8))
        py = int(ly.get('button_padding_y', 4))
        bw = int(ly.get('border_width', 1))
        rh = int(ly.get('row_height', 24))

        default_font = f.get('default', ('Segoe UI', 10))
        bold_font    = f.get('bold',    ('Segoe UI', 10, 'bold'))
        small_font   = f.get('small',   ('Segoe UI', 9))
        mono_font    = f.get('mono',    ('Consolas', 9))

        root.configure(bg=bg)

        def _safe(widget: str, **kw) -> None:
            """Apply style options, silently skipping any rejected by Tcl/Tk."""
            try:
                style.configure(widget, **kw)
            except Exception as _e:
                print(f"[ThemeManager] skipped {widget} option: {_e}")

        def _safe_map(widget: str, **kw) -> None:
            try:
                style.map(widget, **kw)
            except Exception as _e:
                print(f"[ThemeManager] skipped map {widget}: {_e}")

        # ── Global defaults ───────────────────────────────────────────────
        _safe('.', background=bg, foreground=fg,
              font=default_font, borderwidth=0, relief='flat')

        # ── Frame / LabelFrame ────────────────────────────────────────────
        _safe('TFrame',      background=bg)
        _safe('TLabelframe', background=bg, foreground=fg,
              bordercolor=brd, borderwidth=bw, relief='groove')
        _safe('TLabelframe.Label', background=bg, foreground=fg, font=bold_font)

        # ── Label ─────────────────────────────────────────────────────────
        _safe('TLabel',        background=bg, foreground=fg)
        _safe('Status.TLabel', background=sb_bg, foreground=fg_d,
              font=small_font, relief='flat', padding=(4, 2))

        # ── Button ────────────────────────────────────────────────────────
        _safe('TButton',
              background=card, foreground=fg,
              padding=(px, py), relief='flat',
              borderwidth=bw, focuscolor=acc)
        _safe_map('TButton',
                  background=[('active', hov), ('pressed', pre)],
                  bordercolor=[('focus', acc)])

        _safe('Accent.TButton',
              background=acc, foreground='white',
              padding=(px, py), relief='flat', borderwidth=0)
        _safe_map('Accent.TButton',
                  background=[('active', acc_h), ('pressed', acc_p)])

        # ── Entry / Combobox ──────────────────────────────────────────────
        _safe('TEntry',
              fieldbackground=card, foreground=fg,
              insertcolor=acc, bordercolor=brd,
              lightcolor=brd, darkcolor=brd, borderwidth=bw)
        _safe_map('TEntry', bordercolor=[('focus', acc)])

        _safe('TCombobox',
              fieldbackground=card, foreground=fg,
              selectbackground=sel_bg, selectforeground=sel_fg,
              bordercolor=brd)
        _safe_map('TCombobox', fieldbackground=[('readonly', card)])

        # ── Spinbox ───────────────────────────────────────────────────────
        _safe('TSpinbox',
              fieldbackground=card, foreground=fg,
              arrowcolor=fg_d, bordercolor=brd, borderwidth=bw)
        _safe_map('TSpinbox', bordercolor=[('focus', acc)])

        # ── Treeview ──────────────────────────────────────────────────────
        _safe('Treeview',
              background=card, foreground=fg,
              fieldbackground=card, rowheight=rh, borderwidth=0)
        _safe('Treeview.Heading',
              background=th_bg, foreground=fg, font=bold_font, relief='flat')
        _safe_map('Treeview',
                  background=[('selected', sel_bg)],
                  foreground=[('selected', sel_fg)])
        _safe_map('Treeview.Heading', background=[('active', hov)])

        # ── Scrollbar ─────────────────────────────────────────────────────
        _safe('TScrollbar',
              background=trough, troughcolor=bg,
              arrowcolor=fg_d, borderwidth=0, relief='flat')
        _safe_map('TScrollbar', background=[('active', brd)])

        # ── Progressbar ───────────────────────────────────────────────────
        _safe('TProgressbar', troughcolor=trough, background=prog, borderwidth=0)

        # ── Notebook ──────────────────────────────────────────────────────
        _safe('TNotebook', background=bg)
        _safe('TNotebook.Tab', background=trough, foreground=fg, padding=(10, 4))
        _safe_map('TNotebook.Tab',
                  background=[('selected', card), ('active', hov)],
                  foreground=[('selected', acc)])

        # ── Separator ─────────────────────────────────────────────────────
        _safe('TSeparator', background=brd)

        # ── Scale ─────────────────────────────────────────────────────────
        # Only use options that are valid for ttk's Scale element.
        _safe('TScale', background=bg, troughcolor=trough)
        _safe_map('TScale', background=[('active', hov)])

        # ── Checkbutton / Radiobutton ─────────────────────────────────────
        _safe('TCheckbutton', background=bg, foreground=fg)
        _safe('TRadiobutton', background=bg, foreground=fg)

        # ── Panedwindow ───────────────────────────────────────────────────
        _safe('TPanedwindow', background=bg)

        # ── Windows 11 title bar colour (best-effort) ─────────────────────
        self._set_titlebar_color(root, bg)

    @staticmethod
    def _set_titlebar_color(root: tk.Tk, color_hex: str) -> None:
        """Try to match the title bar background to the theme (Windows 11 only)."""
        try:
            import ctypes
            hwnd = int(root.frame(), 16)
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            bgr = r | (g << 8) | (b << 16)     # COLORREF is BGR
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 35,
                ctypes.byref(ctypes.c_int(bgr)), ctypes.sizeof(ctypes.c_int),
            )
        except Exception:
            pass                                # Non-Windows or unsupported build
