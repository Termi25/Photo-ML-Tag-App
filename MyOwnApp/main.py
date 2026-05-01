"""main.py — application entry point.

Run with:
    python main.py

The original file_explorer.py is kept as the authoritative source for
FileExplorer; this module simply provides a clean top-level entry point
that does not pollute the package namespace.
"""

import sys
import io
import os
import traceback
import threading
from datetime import datetime
from pathlib import Path

# Ensure the console can display Unicode (✓ ✗ ⚠) on Windows
try:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from typing import Any as _Any

_TkDnD: _Any = None
DND_FILES: _Any = None
_TKDND_AVAILABLE = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD as _TkDnD  # type: ignore[import-untyped]
    _TKDND_AVAILABLE = True
except ImportError:
    pass

from ui import FileExplorer


def _install_crash_logging(root: object) -> None:
    log_path = Path(os.getcwd()) / 'crash.log'

    def _write(kind: str, text: str) -> None:
        try:
            stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with log_path.open('a', encoding='utf-8', errors='replace') as f:
                f.write(f"\n[{stamp}] {kind}\n")
                f.write(text)
                if not text.endswith('\n'):
                    f.write('\n')
        except Exception:
            pass

    def _tk_report_callback_exception(exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        _write('TkException', ''.join(traceback.format_exception(exc_type, exc, tb)))

    try:
        # Tkinter calls this when a callback raises.
        setattr(root, 'report_callback_exception', _tk_report_callback_exception)
    except Exception:
        pass

    # Background thread unhandled exceptions
    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        _write('ThreadException', ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

    try:
        threading.excepthook = _thread_excepthook  # type: ignore[assignment]
    except Exception:
        pass

    # Main thread unhandled exceptions
    def _sys_excepthook(exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        _write('UnhandledException', ''.join(traceback.format_exception(exc_type, exc, tb)))
        try:
            sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass

    try:
        sys.excepthook = _sys_excepthook  # type: ignore[assignment]
    except Exception:
        pass


def main() -> None:
    if _TKDND_AVAILABLE and _TkDnD is not None:
        root: _Any = _TkDnD.Tk()
    else:
        import tkinter as tk
        root = tk.Tk()

    _install_crash_logging(root)
    app = FileExplorer(root)  # noqa: F841
    root.mainloop()


if __name__ == '__main__':
    main()
