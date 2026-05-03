"""
ML-Enhanced Image Sorting Application
4-tier architecture with supervised/unsupervised learning capabilities
"""

import os
import sys
import io
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Toplevel, Scrollbar, simpledialog
from pathlib import Path
import hashlib
try:
    import lmdb as lmdb  # type: ignore[import-untyped]
except ImportError:
    lmdb = None  # type: ignore[assignment]
import io as _io
import platform
import string
import queue
from PIL import Image, ImageTk, ImageOps
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import json

# Ensure the console can display Unicode characters (e.g. ✓ ✗ ⚠) on Windows
# where the default cp1250/cp850 codec would raise UnicodeEncodeError.
try:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from typing import Any as _Any, Dict, Optional
_TkDnD: _Any = None
DND_FILES: _Any = None
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD as _TkDnD  # type: ignore[import-untyped]
    _TKDND_AVAILABLE = True
except ImportError:
    _TKDND_AVAILABLE = False

_GALLERY_COLS = 5   # target columns in icon/gallery view
_GALLERY_GAP = 4    # pixels of space between cells (each side gets GAP//2)
_PREVIEW_FILENAME   = '.folder_preview.png'   # static preview stored inside each folder
_PREVIEW_RESOLUTION = 768                      # pixel size of the stored square preview
_PREVIEW_ZOOM       = 1.4                      # zoom-in factor applied to each quadrant tile

# Thumbnail cache encoding/versioning. Bump this to invalidate old cached thumbs.
_THUMB_CACHE_VERSION = 3                       # bumped: EXIF orientation fix + WebP format
_THUMB_CACHE_FORMAT  = 'WEBP'                  # lossless-ish compression, ~30% smaller than JPEG
_INBOX_FOLDER       = '_INBOX'                 # drop-folder for new unclassified images
_THUMBS_FOLDER      = 'thumbs'                 # thumbnail cache folder — excluded everywhere

# Import application modules
from config import config_manager, HyperParameters
from data_layer import metadata_store, model_storage
from logic_controller import logic_controller
from ml_module import InferenceEngine, TrainingLoop
from metrics import metrics_collector


_MAX_DETAIL_LINES = 300


class TrainingProgressDialog:
    def __init__(self, parent, title: str = "Training Progress",
                 cancel_callback=None) -> None:
        self._cancel_callback = cancel_callback
        self.window = Toplevel(parent)
        self.window.title(title)
        self.window.geometry("620x430")
        self.window.resizable(True, True)
        self.window.transient(parent)

        self.status_label = ttk.Label(self.window, text="Initializing...",
                                      font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=8)

        metrics_frame = ttk.Frame(self.window)
        metrics_frame.pack(pady=4)
        ttk.Label(metrics_frame, text="Train Acc:", font=('Arial', 11)).grid(row=0, column=0, padx=4)
        self.train_acc_label = ttk.Label(metrics_frame, text="--",
                                         font=('Arial', 11, 'bold'), foreground='blue')
        self.train_acc_label.grid(row=0, column=1, padx=4)
        ttk.Label(metrics_frame, text="Val Acc:", font=('Arial', 11)).grid(row=0, column=2, padx=4)
        self.val_acc_label = ttk.Label(metrics_frame, text="--",
                                       font=('Arial', 11, 'bold'), foreground='green')
        self.val_acc_label.grid(row=0, column=3, padx=4)
        ttk.Label(metrics_frame, text="Gap:", font=('Arial', 11)).grid(row=0, column=4, padx=4)
        self.gap_label = ttk.Label(metrics_frame, text="--", font=('Arial', 11, 'bold'))
        self.gap_label.grid(row=0, column=5, padx=4)

        self.warning_label = ttk.Label(self.window, text="", font=('Arial', 11), foreground='red')
        self.warning_label.pack(pady=1)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.window, variable=self.progress_var,
                                            maximum=100, length=570)
        self.progress_bar.pack(pady=8, padx=25)

        details_frame = ttk.Frame(self.window)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=14,
                                    font=('Consolas', 9), state='disabled')
        _sb = ttk.Scrollbar(details_frame, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=_sb.set)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._detail_line_count = 0

        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=8)
        self._cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._on_cancel,
            state=tk.NORMAL if cancel_callback else tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=6)
        self._close_btn = ttk.Button(btn_frame, text="Close",
                                     command=self.window.destroy, state=tk.DISABLED)
        self._close_btn.pack(side=tk.LEFT, padx=6)

        self.is_complete  = False
        self.is_cancelled = False

        self.window.protocol('WM_DELETE_WINDOW', self._on_window_close)

    def _on_window_close(self) -> None:
        if self.is_complete or self.is_cancelled:
            self.window.destroy()
        else:
            self._on_cancel()

    def _alive(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def _on_cancel(self) -> None:
        self._cancel_btn.config(state=tk.DISABLED)
        if self._cancel_callback:
            self._cancel_callback()

    def update_status(self, message: str) -> None:
        if self._alive():
            self.status_label.config(text=message)

    def update_progress(self, value: float) -> None:
        if self._alive():
            self.progress_var.set(value)

    def add_detail(self, message: str) -> None:
        if not self._alive():
            return
        self.details_text.config(state='normal')
        self.details_text.insert(tk.END, message + "\n")
        self._detail_line_count += 1
        if self._detail_line_count > _MAX_DETAIL_LINES:
            excess = self._detail_line_count - _MAX_DETAIL_LINES
            self.details_text.delete("1.0", f"{excess + 1}.0")
            self._detail_line_count = _MAX_DETAIL_LINES
        self.details_text.config(state='disabled')

    def scroll_to_end(self) -> None:
        if self._alive():
            self.details_text.see(tk.END)

    def update_metrics(self, train_acc: float, val_acc: float, gap: float) -> None:
        if not self._alive():
            return
        self.train_acc_label.config(text=f"{train_acc:.1f}%")
        self.val_acc_label.config(text=f"{val_acc:.1f}%")
        self.gap_label.config(text=f"{gap:.1f}%")
        if gap > 15:
            self.gap_label.config(foreground='red')
            self.warning_label.config(text="⚠ Overfitting detected")
        elif gap > 10:
            self.gap_label.config(foreground='orange')
            self.warning_label.config(text="")
        else:
            self.gap_label.config(foreground='green')
            self.warning_label.config(text="")

    def mark_complete(self, success: bool = True) -> None:
        self.is_complete = True
        if not self._alive():
            return
        self.status_label.config(
            text="✓ Training Completed Successfully!" if success else "✗ Training Failed")
        if success:
            self.progress_var.set(100)
        self._cancel_btn.config(state=tk.DISABLED)
        self._close_btn.config(state=tk.NORMAL)

    def mark_cancelled(self) -> None:
        self.is_cancelled = True
        if not self._alive():
            return
        self.status_label.config(text="Training cancelled.")
        self._cancel_btn.config(state=tk.DISABLED)
        self._close_btn.config(state=tk.NORMAL)

    def show(self) -> None:
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth()  // 2) - 310
        y = (self.window.winfo_screenheight() // 2) - 215
        self.window.geometry(f'620x430+{x}+{y}')


class SplashScreen:
    """Startup loading screen — blocks all user interaction until the app is ready.

    Shown as a borderless, centered Toplevel with grab_set() so nothing behind
    it is reachable.  Call ``dismiss()`` on the UI thread to remove it.
    """

    _W, _H   = 440, 252
    _BG      = '#1e1e2e'
    _FG_HEAD = '#cdd6f4'
    _FG_SUB  = '#89b4fa'
    _FG_STEP = '#a6e3a1'
    _FG_DIM  = '#6c7086'
    _FG_BORD = '#313244'

    def __init__(self, parent: tk.Tk) -> None:
        self._parent    = parent
        self._dismissed = False

        win = tk.Toplevel(parent)
        self.window = win
        win.overrideredirect(True)
        win.resizable(False, False)

        # Centre over the parent window
        parent.update_idletasks()
        pw = parent.winfo_width()  or 1200
        ph = parent.winfo_height() or 800
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        x  = px + (pw - self._W) // 2
        y  = py + (ph - self._H) // 2
        win.geometry(f'{self._W}x{self._H}+{x}+{y}')

        # Outer border frame (single-pixel accent border)
        border = tk.Frame(win, bg=self._FG_BORD, padx=1, pady=1)
        border.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(border, bg=self._BG)
        inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(inner, text='ML Image Sorter', bg=self._BG, fg=self._FG_HEAD,
                 font=('Arial', 20, 'bold')).pack(pady=(26, 2))
        tk.Label(inner, text='Supervised image categorisation with ML',
                 bg=self._BG, fg=self._FG_SUB, font=('Arial', 9)).pack()

        self._status_var = tk.StringVar(value='Starting…')
        tk.Label(inner, textvariable=self._status_var,
                 bg=self._BG, fg=self._FG_STEP,
                 font=('Consolas', 9), wraplength=410,
                 justify='center').pack(pady=(16, 8))

        pb_wrap = tk.Frame(inner, bg=self._BG)
        pb_wrap.pack(fill=tk.X, padx=32)
        self._pb = ttk.Progressbar(pb_wrap, mode='indeterminate', length=376)
        self._pb.pack(fill=tk.X)
        self._pb.start(10)

        tk.Label(inner, text='Please wait…',
                 bg=self._BG, fg=self._FG_DIM, font=('Arial', 8)).pack(pady=(14, 0))

        # Grab all pointer + keyboard input
        win.grab_set()
        win.lift()
        try:
            win.attributes('-topmost', True)
        except Exception:
            pass

    # ------------------------------------------------------------------

    def set_status(self, msg: str) -> None:
        """Update the status line.  Must be called on the UI thread."""
        if self._dismissed:
            return
        try:
            self._status_var.set(msg)
            self.window.update_idletasks()
        except Exception:
            pass

    def dismiss(self) -> None:
        """Release the grab and destroy the splash.  Call on the UI thread."""
        if self._dismissed:
            return
        self._dismissed = True
        try:
            self._pb.stop()
        except Exception:
            pass
        try:
            self.window.grab_release()
        except Exception:
            pass
        try:
            self.window.destroy()
        except Exception:
            pass


class MoveProgressDialog:
    """Non-blocking progress dialog for file move / copy operations.

    Created on the UI thread; updated from background threads via ``root.after``.
    Does NOT call ``grab_set()`` so the rest of the UI stays responsive.
    """

    def __init__(self, parent, title: str = "Moving Files…", total: int = 0):
        self.window = Toplevel(parent)
        self.window.title(title)
        self.window.geometry("500x230")
        self.window.resizable(False, False)
        self.window.transient(parent)
        # Intentionally NOT modal — no grab_set()

        self._total = max(total, 1)

        self.status_label = ttk.Label(self.window, text="Preparing…",
                                      font=('Arial', 11, 'bold'))
        self.status_label.pack(pady=(12, 2), padx=16, anchor='w')

        self.file_label = ttk.Label(self.window, text="",
                                    font=('Arial', 10), foreground='gray',
                                    wraplength=460, justify='left')
        self.file_label.pack(padx=16, anchor='w')

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.window,
                                            variable=self.progress_var,
                                            maximum=100, length=460)
        self.progress_bar.pack(pady=10, padx=16)

        counter_row = ttk.Frame(self.window)
        counter_row.pack(fill=tk.X, padx=16)
        self.counter_label = ttk.Label(counter_row,
                                       text=f"0 / {self._total}",
                                       font=('Arial', 10))
        self.counter_label.pack(side=tk.LEFT)
        self.error_label = ttk.Label(counter_row, text="",
                                     font=('Arial', 10), foreground='red')
        self.error_label.pack(side=tk.RIGHT)

        self.close_btn = ttk.Button(self.window, text="Close",
                                    command=self.window.destroy,
                                    state=tk.DISABLED)
        self.close_btn.pack(pady=8)

        # Centre on screen
        self.window.update_idletasks()
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        self.window.geometry(f'500x230+{(sw-500)//2}+{(sh-230)//2}')

    # --- called from background thread via root.after ---

    def update(self, current_name: str, done: int, errors: int = 0) -> None:
        """Update progress bar and labels (must be called via root.after)."""
        try:
            pct = min(100.0, done / self._total * 100)
            self.progress_var.set(pct)
            self.file_label.config(text=current_name)
            self.counter_label.config(text=f"{done} / {self._total}")
            if errors:
                self.error_label.config(text=f"{errors} error(s)")
        except Exception:
            pass

    def mark_done(self, moved: int, skipped: int = 0) -> None:
        """Mark the operation complete and enable the Close button."""
        try:
            label = f"✓ Done — {moved} moved"
            if skipped:
                label += f", {skipped} skipped"
            self.status_label.config(text=label)
            self.file_label.config(text="")
            self.progress_var.set(100)
            self.counter_label.config(text=f"{moved} / {self._total}")
            self.close_btn.config(state=tk.NORMAL)
        except Exception:
            pass

    def mark_error(self, message: str) -> None:
        """Mark the operation as failed."""
        try:
            self.status_label.config(text="✗ Error occurred", foreground='red')
            self.error_label.config(text=message[:70])
            self.close_btn.config(state=tk.NORMAL)
        except Exception:
            pass


class InboxTrainingDialog:
    """Dialog for manually tagging inbox images and fine-tuning the existing model.

    Layout
    ------
    Left  : scrollable image list with per-image tag counts.
    Right : large preview + tag editor (add / remove) + known-tags browser.
    Bottom: epoch spinner and "Fine-Tune Model" button.

    The dialog pre-populates each image with model predictions (if a model is
    loaded) so the user only needs to correct/confirm, not start from scratch.
    """

    _THUMB_W    = 80
    _THUMB_H    = 60
    _PREVIEW_SZ = 280
    _IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif'}

    def __init__(self, parent, inbox_path: Path, controller, known_tags: list):
        self.parent       = parent
        self.inbox_path   = inbox_path
        self.controller   = controller
        self.known_tags   = sorted(known_tags)

        # annotations: {str(image_path): [tag, ...]}
        self.annotations: dict = {}

        # Gather inbox images (exclude the folder-preview thumbnail)
        self.images: list[Path] = sorted(
            f for f in inbox_path.iterdir()
            if f.is_file() and f.suffix.lower() in self._IMG_EXTS
            and f.name != _PREVIEW_FILENAME and not f.name.startswith('.')
        )

        self._thumb_refs:  list = []   # keep PhotoImage refs alive
        self._preview_ref         = None
        self._selected_idx: int   = -1

        # ---- Window ----------------------------------------------------------
        self.window = Toplevel(parent)
        self.window.title(
            f"Inbox: Tag & Fine-Tune  ({len(self.images)} image(s))")
        self.window.geometry('960x620')
        self.window.resizable(True, True)
        self.window.transient(parent)
        self.window.grab_set()

        # Cancel flag for background pre-population work.
        # (Do not call Tk APIs from worker threads; use this flag instead.)
        self._pred_cancel = threading.Event()
        try:
            self.window.bind('<Destroy>', lambda _e: self._pred_cancel.set())
        except Exception:
            pass

        self._build_ui()
        self._populate_image_list()
        if self.images:
            self._select_image(0)

        # Centre on screen
        self.window.update_idletasks()
        sw = self.window.winfo_screenwidth()
        sh = self.window.winfo_screenheight()
        self.window.geometry(f'960x620+{(sw-960)//2}+{(sh-620)//2}')

        # ---- Pre-populate with model suggestions AFTER dialog is visible ----
        # Deferred so the dialog renders immediately and the main thread stays
        # responsive while inference runs image-by-image.
        if controller.tagging_engine.model_loaded and self.images:
            self.window.after(100, self._prepopulate_predictions)

    # ------------------------------------------------------------------
    # Pre-population (deferred to keep UI responsive)
    # ------------------------------------------------------------------

    def _prepopulate_predictions(self):
        """Run model predictions for all inbox images and pre-fill annotations.
        Called via window.after() so the dialog is already visible."""
        if not self.controller.tagging_engine.model_loaded:
            return
        # IMPORTANT: model inference can be slow. Running it on the Tk thread
        # freezes the whole UI, so do the work on a background thread and only
        # post UI updates back via `after()`.

        # Avoid starting multiple concurrent pre-population workers.
        try:
            if getattr(self, '_pred_inflight', False):
                return
            self._pred_inflight = True
        except Exception:
            return

        def _worker():
            try:
                for img in self.images:
                    if self._pred_cancel.is_set():
                        return
                    try:
                        preds = self.controller.tagging_engine.predict_tags_for_image(str(img))
                        if preds:
                            key = str(img)
                            if key not in self.annotations:
                                self.annotations[key] = [p['tag_name'] for p in preds[:5]]
                    except Exception:
                        pass
            finally:
                def _apply():
                    try:
                        self._pred_inflight = False
                    except Exception:
                        pass
                    # Dialog may have been closed while we were working.
                    try:
                        if not self.window.winfo_exists():
                            return
                    except Exception:
                        return
                    # Refresh UI after predictions are loaded
                    try:
                        self._populate_image_list()
                        if self._selected_idx >= 0:
                            self._update_tag_display()
                        self._update_bottom_status()
                    except Exception:
                        pass

                try:
                    self.window.after(0, _apply)
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Instruction bar -------------------------------------------------
        info = ttk.Label(
            self.window,
            text=("Tag each inbox image, then press 'Fine-Tune Model'. "
                  "Only tags the model already knows are used — "
                  "unknown tags are shown in orange as a reminder."),
            font=('Arial', 9), foreground='gray', wraplength=940)
        info.pack(fill=tk.X, padx=8, pady=(6, 2))

        ttk.Separator(self.window, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ---- Main paned area -------------------------------------------------
        pane = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- Left panel: image list ------------------------------------------
        left = ttk.Frame(pane, width=230)
        pane.add(left, weight=1)

        ttk.Label(left, text='Images', font=('Arial', 10, 'bold')).pack(
            anchor='w', padx=4, pady=(2, 0))

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._img_listbox = tk.Listbox(
            list_frame, font=('Consolas', 9),
            selectmode=tk.SINGLE, activestyle='none',
            selectbackground='#4a90d9', selectforeground='white')
        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                             command=self._img_listbox.yview)
        self._img_listbox.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._img_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._img_listbox.bind('<<ListboxSelect>>', self._on_list_select)

        # ---- Right panel: preview + tag editor --------------------------------
        right = ttk.Frame(pane)
        pane.add(right, weight=3)

        # Preview canvas
        self._canvas = Canvas(right, width=self._PREVIEW_SZ,
                               height=self._PREVIEW_SZ, bg='#2b2b2b',
                               highlightthickness=0)
        self._canvas.pack(pady=(4, 2))

        # Current image name
        self._img_name_var = tk.StringVar(value='')
        ttk.Label(right, textvariable=self._img_name_var,
                  font=('Arial', 9, 'bold')).pack()

        # Active-tags row (chips)
        ttk.Label(right, text='Tags:', font=('Arial', 9, 'bold')).pack(
            anchor='w', padx=8, pady=(4, 0))
        self._chips_frame = ttk.Frame(right)
        self._chips_frame.pack(fill=tk.X, padx=8, pady=2)

        # Add-tag row
        add_row = ttk.Frame(right)
        add_row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(add_row, text='Add tag:', font=('Arial', 9)).pack(
            side=tk.LEFT)
        self._tag_var = tk.StringVar()
        self._tag_entry = ttk.Entry(add_row, textvariable=self._tag_var,
                                     width=22)
        self._tag_entry.pack(side=tk.LEFT, padx=4)
        self._tag_entry.bind('<Return>', lambda e: self._add_tag_from_entry())
        ttk.Button(add_row, text='+Add',
                   command=self._add_tag_from_entry).pack(side=tk.LEFT)

        # Known-tags browser
        ttk.Label(right, text='Known tags (double-click to add):',
                  font=('Arial', 9)).pack(anchor='w', padx=8, pady=(6, 0))
        kt_frame = ttk.Frame(right)
        kt_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        self._kt_var = tk.StringVar(value=self.known_tags)  # type: ignore[arg-type]
        self._kt_listbox = tk.Listbox(
            kt_frame, listvariable=self._kt_var,
            font=('Consolas', 9), height=6,
            selectmode=tk.SINGLE, exportselection=False)
        kt_vsb = ttk.Scrollbar(kt_frame, orient=tk.VERTICAL,
                                command=self._kt_listbox.yview)
        self._kt_listbox.configure(yscrollcommand=kt_vsb.set)
        kt_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._kt_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._kt_listbox.bind('<Double-Button-1>', self._add_tag_from_browser)

        # Live filter for known-tags browser
        filter_row = ttk.Frame(right)
        filter_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(filter_row, text='Filter:', font=('Arial', 9)).pack(
            side=tk.LEFT)
        self._filter_var = tk.StringVar()
        self._filter_var.trace_add('write', self._filter_known_tags)
        ttk.Entry(filter_row, textvariable=self._filter_var,
                  width=20).pack(side=tk.LEFT, padx=4)

        # ---- Bottom bar: epoch spinner + action buttons ----------------------
        ttk.Separator(self.window, orient=tk.HORIZONTAL).pack(fill=tk.X)
        bot = ttk.Frame(self.window)
        bot.pack(fill=tk.X, padx=8, pady=6)

        self._status_var = tk.StringVar(value='')
        ttk.Label(bot, textvariable=self._status_var,
                  font=('Arial', 9), foreground='gray').pack(side=tk.LEFT)

        ttk.Label(bot, text='  Epochs:', font=('Arial', 9)).pack(side=tk.LEFT)
        self._epochs_var = tk.IntVar(value=5)
        ttk.Spinbox(bot, from_=1, to=20, width=4,
                    textvariable=self._epochs_var).pack(side=tk.LEFT, padx=4)

        self._ft_btn = ttk.Button(
            bot, text='Fine-Tune Model (0 labeled)',
            command=self._start_fine_tune)
        self._ft_btn.pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(bot, text='Close',
                   command=self.window.destroy).pack(side=tk.RIGHT)

        self._update_bottom_status()

    # ------------------------------------------------------------------
    # Image list helpers
    # ------------------------------------------------------------------

    def _populate_image_list(self):
        self._img_listbox.delete(0, tk.END)
        for img in self.images:
            n = len(self.annotations.get(str(img), []))
            mark = '✓' if n > 0 else '○'
            self._img_listbox.insert(
                tk.END, f' {mark} {img.name}  [{n}]')

    def _refresh_list_item(self, idx: int):
        """Refresh a single row without rebuilding the whole list."""
        img = self.images[idx]
        n   = len(self.annotations.get(str(img), []))
        mark = '✓' if n > 0 else '○'
        self._img_listbox.delete(idx)
        self._img_listbox.insert(idx, f' {mark} {img.name}  [{n}]')
        self._img_listbox.selection_set(idx)

    def _on_list_select(self, _event):
        sel = self._img_listbox.curselection()
        if sel:
            self._select_image(int(sel[0]))

    def _select_image(self, idx: int):
        self._selected_idx = idx
        img_path = self.images[idx]
        self._img_name_var.set(img_path.name)

        # Thumbnail preview
        try:
            pil_img = Image.open(str(img_path))
            try:
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass
            pil_img = pil_img.convert('RGB')
            pil_img.thumbnail((self._PREVIEW_SZ, self._PREVIEW_SZ),
                              Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            self._preview_ref = photo
            self._canvas.delete('all')
            cx = self._PREVIEW_SZ // 2
            cy = self._PREVIEW_SZ // 2
            self._canvas.create_image(cx, cy, image=photo, anchor='center')
        except Exception:
            self._canvas.delete('all')
            self._canvas.create_text(
                self._PREVIEW_SZ // 2, self._PREVIEW_SZ // 2,
                text='(preview unavailable)', fill='gray')

        self._update_tag_display()

    # ------------------------------------------------------------------
    # Tag management
    # ------------------------------------------------------------------

    def _update_tag_display(self):
        """Rebuild the chip row for the currently selected image."""
        for w in self._chips_frame.winfo_children():
            w.destroy()

        if self._selected_idx < 0:
            return

        img_key = str(self.images[self._selected_idx])
        tags    = list(self.annotations.get(img_key, []))

        for tag in tags:
            # Highlight unknown tags in orange
            colour = 'orange' if tag not in self.known_tags else '#4a90d9'
            chip = tk.Frame(self._chips_frame, bg=colour,
                            bd=0, padx=4, pady=2)
            chip.pack(side=tk.LEFT, padx=2, pady=2)
            tk.Label(chip, text=tag, bg=colour, fg='white',
                     font=('Arial', 9)).pack(side=tk.LEFT)
            # × button captures current tag value in default arg
            tk.Button(chip, text='×', bg=colour, fg='white',
                      font=('Arial', 8, 'bold'), bd=0, cursor='hand2',
                      command=lambda t=tag: self._remove_tag(t)).pack(
                          side=tk.LEFT)

        if not tags:
            ttk.Label(self._chips_frame, text='(no tags)',
                      foreground='gray', font=('Arial', 9)).pack(
                          side=tk.LEFT)

    def _add_tag_from_entry(self):
        tag = self._tag_var.get().strip()
        if tag:
            self._add_tag(tag)
            self._tag_var.set('')

    def _add_tag_from_browser(self, _event=None):
        sel = self._kt_listbox.curselection()
        if sel:
            tag = self._kt_listbox.get(int(sel[0]))
            self._add_tag(tag)

    def _add_tag(self, tag: str):
        if self._selected_idx < 0 or not tag:
            return
        img_key = str(self.images[self._selected_idx])
        tags = self.annotations.setdefault(img_key, [])
        if tag not in tags:
            tags.append(tag)
        self._update_tag_display()
        self._refresh_list_item(self._selected_idx)
        self._update_bottom_status()

    def _remove_tag(self, tag: str):
        if self._selected_idx < 0:
            return
        img_key = str(self.images[self._selected_idx])
        tags = self.annotations.get(img_key, [])
        if tag in tags:
            tags.remove(tag)
        if not tags:
            self.annotations.pop(img_key, None)
        self._update_tag_display()
        self._refresh_list_item(self._selected_idx)
        self._update_bottom_status()

    def _filter_known_tags(self, *_):
        q = self._filter_var.get().strip().lower()
        filtered = [t for t in self.known_tags if q in t.lower()] \
            if q else self.known_tags
        self._kt_var.set(filtered)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Bottom status
    # ------------------------------------------------------------------

    def _update_bottom_status(self):
        labeled = sum(
            1 for v in self.annotations.values() if v)
        self._status_var.set(
            f'{labeled} of {len(self.images)} image(s) labeled')
        self._ft_btn.config(
            text=f'Fine-Tune Model ({labeled} labeled)',
            state=tk.NORMAL if labeled > 0 else tk.DISABLED)

    # ------------------------------------------------------------------
    # Fine-tune trigger
    # ------------------------------------------------------------------

    def _start_fine_tune(self):
        labeled = {k: v for k, v in self.annotations.items() if v}
        if not labeled:
            messagebox.showwarning('No labels',
                'Please tag at least one image before fine-tuning.',
                parent=self.window)
            return

        epochs = max(1, self._epochs_var.get())
        if not messagebox.askyesno(
                'Fine-Tune Model',
                f'Fine-tune the model on {len(labeled)} labeled image(s) '
                f'for {epochs} epoch(s)?\n\n'
                'This runs in the background. The current model will be '
                'updated when finished.',
                parent=self.window):
            return

        # Queue fine-tune task (non-blocking)
        self.controller.fine_tune_from_inbox_labels(labeled, epochs=epochs)
        self._status_var.set(
            f'Fine-tuning queued ({len(labeled)} image(s), {epochs} epoch(s))…')
        self._ft_btn.config(state=tk.DISABLED)
        messagebox.showinfo(
            'Queued',
            'Fine-tuning has been queued and will run in the background.\n'
            'Check the status bar for progress.',
            parent=self.window)


class _TrainingState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = None
        self._progress = None
        self._metrics = None
        self._details: list = []
        self._terminal = None

    def push(self, kind: str, value) -> None:
        with self._lock:
            if kind == 'status':
                self._status = value
            elif kind == 'progress':
                self._progress = value
            elif kind == 'metrics':
                self._metrics = value
            elif kind == 'detail':
                if len(self._details) < 500:
                    self._details.append(str(value))
            elif kind in ('complete', 'error', 'cancelled'):
                self._terminal = (kind, value)

    def drain(self):
        with self._lock:
            out = (self._status, self._progress, self._metrics,
                   list(self._details), self._terminal)
            self._status   = None
            self._progress = None
            self._metrics  = None
            self._details.clear()
            self._terminal = None
        return out


class FileExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("ML-Enhanced Image Sorting Application")
        self.root.geometry("1400x800")
        
        # Default to the included ExampleImageFolder when present, fall back to home dir
        default_folder = Path(__file__).parent.parent / "ExampleImageFolder"
        if default_folder.exists():
            self.current_path = default_folder
        else:
            self.current_path = Path.home()
        # root_folder is fixed to the app directory so the DB and ML_Training_Data
        # always land next to the source files, never inside a browsed image folder.
        self.root_folder = Path(__file__).parent.parent

        # App state file (per-root) — stores preview snapshots and UI state
        self._app_state_path = self.root_folder / '.app_state.json'
        self._app_state: dict = {}
        self._load_app_state()
        self._ui_state: dict = self._app_state.get('ui', {})
        # Dirty-flag + debounce handle for _save_app_state
        self._app_state_dirty: bool = False
        self._app_state_save_job: str | None = None
        # Predeclare dynamic UI attributes used before all tabs are created.
        self._thumbs_running_gen = None
        self._view_opts_content = None
        self._view_opts_toggle = None
        self._view_opts_expanded = tk.BooleanVar(value=True)
        self._cols_var = tk.IntVar(value=_GALLERY_COLS)
        self._file_sort_var = tk.StringVar(value='name')
        # Optional override for gallery thumbnail height (persisted in UI state)
        self._view_zoom_override: Optional[int] = None
        try:
            if isinstance(self._ui_state, dict):
                raw_gallery_size = self._ui_state.get('gallery_size')
                if raw_gallery_size is not None:
                    self._view_zoom_override = int(raw_gallery_size)
        except Exception:
            self._view_zoom_override = None
        self._path_var = tk.StringVar(value='')
        self._search_mode_var = tk.StringVar(value='simple')
        self._simple_search_var = tk.StringVar(value='')
        self._tag_display_mode_var = tk.StringVar(value='list')

        # Disk-backed thumbnail cache directory (per root folder)
        self.disk_thumbnail_dir = self.root_folder / '.thumbs'
        try:
            self.disk_thumbnail_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort: ignore if we cannot create cache dir
            pass

        # LMDB environment for thumbnail blobs (fast key-value store)
        try:
            if lmdb is not None:
                lmdb_path = self.root_folder / '.thumbs.lmdb'
                self._lmdb_env = lmdb.open(str(lmdb_path), map_size=1024**3, max_dbs=1)
            else:
                self._lmdb_env = None
        except Exception:
            self._lmdb_env = None
        
        # View mode: 'detailed' or 'icons'
        self.view_mode = 'detailed'

        self._nav_forward_stack: list[Path] = []

        # Show / hide hidden items (OS hidden attribute + app-hidden flag)
        self.show_hidden_var = tk.BooleanVar(value=False)

        # Store image references to prevent garbage collection
        self.image_references = []
        
        # Current selected image for tagging
        self.selected_image_path = None
        self.selected_image_tags = []
        
        # Initialize logic controller
        self.controller = logic_controller

        # Persistent executors for thumbnail and folder work to avoid
        # creating/destroying ThreadPoolExecutors per-load (reduces thrash)
        try:
            cpu = os.cpu_count() or 4
            thumb_workers = min(4, max(1, cpu // 2))
        except Exception:
            thumb_workers = 4
        self._thumb_executor = ThreadPoolExecutor(max_workers=thumb_workers)
        self._folder_executor = ThreadPoolExecutor(max_workers=2)

        # Theme manager — load the saved (or default) theme before building widgets
        from ui.theme import ThemeManager
        self._theme_manager = ThemeManager()
        _saved = config_manager.app_config.__dict__.get('ui_theme', 'light')
        try:
            self._theme_manager.load(_saved)
        except FileNotFoundError:
            self._theme_manager.load('light')
        self._apply_theme()

        # Setup UI
        self.setup_ui()
        
        # Defer expensive startup work until Tk has painted the first frame.
        self.root.after_idle(self._startup_sequence)

        # Save state on close
        try:
            self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        except Exception:
            pass

    def _schedule_path_navigation(self, *_):
        """Debounce path-entry navigation."""
        try:
            job = getattr(self, '_path_nav_job', None)
            if job is not None:
                self.root.after_cancel(job)
        except Exception:
            pass

        def _go():
            try:
                # Use the visible `path_var` (entry) rather than internal _path_var
                if getattr(self, 'path_var', None) is None:
                    return
                path = Path(self.path_var.get().strip())
                if path.exists() and path.is_dir() and path != self.current_path:
                    self.history.append(self.current_path)
                    self._nav_forward_stack.clear()
                    self.load_directory(path)
            except Exception:
                pass

        self._path_nav_job = self.root.after(700, _go)

    def _search_files_by_name_async(self) -> None:
        """Run a filename/path search in the background and display matches."""
        query = getattr(self, '_simple_search_var', tk.StringVar(value='')).get().strip().lower()
        if not query:
            return

        base = self.current_path
        current_gen = self._nav_generation

        def _worker() -> None:
            matches: list[Path] = []
            try:
                for p in base.rglob('*'):
                    if self._nav_generation != current_gen:
                        return
                    if p.name.startswith('.'):
                        continue
                    if any(part.lower() == _THUMBS_FOLDER for part in p.relative_to(base).parts):
                        continue
                    if p.is_file() and p.suffix.lower() in self._IMAGE_EXTS and query in p.name.lower():
                        matches.append(p)
            except Exception:
                matches = []

            def _show() -> None:
                if self._nav_generation != current_gen:
                    return
                self._simple_search_results = matches
                self._search_mode_var.set('simple')
                self.status_var.set(f'Simple search: {len(matches)} match(es)')
                if self.view_mode == 'detailed':
                    self.load_detailed_view([], matches)
                else:
                    self.load_icons_view([], matches)

            self.root.after(0, _show)

        threading.Thread(target=_worker, daemon=True).start()

    def _clear_simple_search(self) -> None:
        self._simple_search_var.set('')
        self._simple_search_results = []
        for v in getattr(self, '_tag_matrix_vars', {}).values():
            v.set(False)
        self.load_directory(self.current_path)

    def go_forward(self):
        """Go to next directory in history."""
        if self._nav_forward_stack:
            next_path = self._nav_forward_stack.pop()
            self.history.append(self.current_path)
            self.load_directory(next_path)

    def _on_close(self):
        try:
            self._save_app_state()
        except Exception:
            pass
        # Shutdown persistent executors
        try:
            with getattr(self, '_thumb_lock', threading.RLock()):
                if getattr(self, '_thumb_executor', None) is not None:
                    try:
                        self._thumb_executor.shutdown(wait=False)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            with getattr(self, '_thumb_lock', threading.RLock()):
                if getattr(self, '_folder_executor', None) is not None:
                    try:
                        self._folder_executor.shutdown(wait=False)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _load_app_state(self) -> None:
        try:
            if self._app_state_path.exists():
                with open(self._app_state_path, 'r', encoding='utf-8') as f:
                    self._app_state = json.load(f)
            else:
                self._app_state = {'folder_previews': {}, 'ui': {}}
        except Exception:
            self._app_state = {'folder_previews': {}, 'ui': {}}

    def _save_app_state(self) -> None:
        try:
            # Update UI state
            ui = self._app_state.get('ui', {})
            ui['view_mode'] = self.view_mode
            ui['supervised_mode'] = bool(config_manager.app_config.supervised_mode)
            ui['gallery_cols'] = getattr(self, '_view_gallery_cols', None)
            ui['gallery_zoom'] = getattr(self, '_view_zoom_override', 0)
            ui['file_sort'] = getattr(self, '_file_sort_var', tk.StringVar(value='name')).get()
            ui['search_sort'] = getattr(self, '_sort_var', tk.StringVar(value='name')).get()
            ui['search_sort_asc'] = bool(getattr(self, '_sort_asc_var', tk.BooleanVar(value=True)).get())
            ui['search_mode'] = getattr(self, '_search_mode_var', tk.StringVar(value='simple')).get()
            ui['simple_search'] = getattr(self, '_simple_search_var', tk.StringVar(value='')).get()
            ui['advanced_search'] = {
                'query': getattr(self, 'search_var', tk.StringVar(value='')).get(),
                'tag_states': getattr(self, '_tag_states', {}),
            }
            ui['tag_display_mode'] = getattr(self, '_tag_display_mode_var', tk.StringVar(value='list')).get()
            try:
                ui['control_tab'] = int(self._control_tabs.index(self._control_tabs.select()))
            except Exception:
                pass
            self._app_state['ui'] = ui

            tmp = self._app_state_path.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._app_state, f, indent=2)
            try:
                tmp.replace(self._app_state_path)
            except Exception:
                tmp.unlink(missing_ok=True)
        except Exception:
            pass

    def _mark_app_state_dirty(self) -> None:
        """Mark app state as needing a save and schedule a debounced flush.

        Safe to call from the UI thread only (background threads must use
        ``root.after(0, self._mark_app_state_dirty)`` to hop to the UI thread).
        A save is coalesced: multiple calls within 2 s produce a single write.
        """
        self._app_state_dirty = True
        if self._app_state_save_job is None:
            self._app_state_save_job = self.root.after(2000, self._flush_app_state)

    def _flush_app_state(self) -> None:
        """Write app state to disk if the dirty flag is set (UI thread only)."""
        self._app_state_save_job = None
        if self._app_state_dirty:
            self._app_state_dirty = False
            self._save_app_state()

    def _restore_ui_state(self) -> None:
        """Restore saved UI selections after the widgets have been created."""
        ui = self._app_state.get('ui', {})
        try:
            if hasattr(self, '_view_gallery_cols') and ui.get('gallery_cols'):
                self._view_gallery_cols = int(ui['gallery_cols'])
                if hasattr(self, '_cols_var'):
                    self._cols_var.set(self._view_gallery_cols)
            if hasattr(self, '_view_zoom_override') and ui.get('gallery_zoom') is not None:
                self._view_zoom_override = int(ui.get('gallery_zoom') or 0)
                if hasattr(self, '_zoom_var'):
                    self._zoom_var.set(self._view_zoom_override)
            if hasattr(self, '_file_sort_var') and ui.get('file_sort'):
                self._file_sort_var.set(ui['file_sort'])
            if hasattr(self, '_sort_var') and ui.get('search_sort'):
                self._sort_var.set(ui['search_sort'])
            if hasattr(self, '_sort_asc_var') and ui.get('search_sort_asc') is not None:
                self._sort_asc_var.set(bool(ui.get('search_sort_asc')))
            if hasattr(self, '_search_mode_var') and ui.get('search_mode'):
                self._search_mode_var.set(ui['search_mode'])
            if hasattr(self, '_simple_search_var') and ui.get('simple_search'):
                self._simple_search_var.set(ui['simple_search'])
            if hasattr(self, '_tag_display_mode_var') and ui.get('tag_display_mode'):
                self._tag_display_mode_var.set(ui['tag_display_mode'])
            adv = ui.get('advanced_search', {})
            if hasattr(self, 'search_var') and adv.get('query'):
                self.search_var.set(adv['query'])
            if hasattr(self, '_tag_states') and isinstance(adv.get('tag_states'), dict):
                self._tag_states = dict(adv['tag_states'])
                if hasattr(self, '_rebuild_chips'):
                    self._rebuild_chips()
            if hasattr(self, '_control_tabs') and ui.get('control_tab') is not None:
                try:
                    self._control_tabs.select(int(ui['control_tab']))
                except Exception:
                    pass
        except Exception:
            pass
        
    # ------------------------------------------------------------------
    # Theme (XML-driven via ui.theme.ThemeManager)
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply the currently loaded theme to the root window."""
        self._theme_manager.apply(self.root)
        # Expose colour shortcuts used throughout the class
        c = self._theme_manager._colors
        self._ACCENT  = c.get('accent',       '#0078d4')
        self._ACCENT_H = c.get('accent_hover', '#106ebe')
        self._BG      = c.get('background',    '#f3f3f3')
        self._SIDEBAR = c.get('sidebar',       '#f0f0f0')
        self._CARD    = c.get('card',          '#ffffff')
        self._FG      = c.get('foreground',    '#1a1a1a')
        self._FG_DIM  = c.get('foreground_dim','#555555')
        self._BORDER  = c.get('border',        '#d0d0d0')

    def change_theme(self, theme_name: str) -> None:
        """Switch to a different XML theme at runtime."""
        try:
            self._theme_manager.load(theme_name)
        except FileNotFoundError as exc:
            from tkinter import messagebox
            messagebox.showerror('Theme Error', str(exc))
            return
        self._apply_theme()
        # Force a repaint of all open frames
        self.root.update_idletasks()
        self.refresh()
        # Persist choice in config
        config_manager.app_config.__dict__['ui_theme'] = theme_name
        config_manager.save_config()

    def setup_ui(self):
        """Setup the user interface with ML components"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Set Root Folder", command=self.set_root_folder_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Clean Orphan Databases", command=self.clean_orphan_databases_action)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="View Metrics", command=self.show_metrics)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Show Hidden Items",
                                  variable=self.show_hidden_var,
                                  command=self.refresh)
        view_menu.add_separator()
        view_menu.add_command(label="Reset Folder Previews", command=self.reset_folder_previews)

        # Theme submenu — one entry per discovered XML file
        view_menu.add_separator()
        theme_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        for _t in self._theme_manager.available_themes():
            theme_menu.add_command(
                label=_t.replace('_', ' ').title(),
                command=lambda t=_t: self.change_theme(t),
            )

        # ── Always-visible navigation toolbar ────────────────────────────────
        nav_toolbar = ttk.Frame(self.root)
        nav_toolbar.pack(side=tk.TOP, fill=tk.X, padx=3, pady=(2, 0))
        self._build_navigation_controls(nav_toolbar)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(side=tk.TOP, fill=tk.X, padx=3, pady=(1, 0))

        # ── Tabbed control strip (Search / Advanced Search / View / ML) ───────
        style = ttk.Style()
        try:
            style.configure('Control.TNotebook.Tab', padding=(6, 2))
        except Exception:
            pass
        # Place the Notebook inside a container so we can resize it to the
        # active tab's needed height rather than sizing to the largest tab.
        control_tabs_container = ttk.Frame(self.root)
        control_tabs_container.pack(side=tk.TOP, fill=tk.X, padx=3, pady=(0, 0))
        # We'll control the container height explicitly when tabs change.
        control_tabs_container.pack_propagate(False)

        control_tabs = ttk.Notebook(control_tabs_container, style='Control.TNotebook')
        control_tabs.pack(side=tk.TOP, fill=tk.X)
        self._control_tabs = control_tabs
        self._control_tabs_container = control_tabs_container

        simple_tab   = ttk.Frame(control_tabs)
        advanced_tab = ttk.Frame(control_tabs)
        view_tab     = ttk.Frame(control_tabs)
        ml_tab       = ttk.Frame(control_tabs)
        control_tabs.add(simple_tab,   text='Search')
        control_tabs.add(advanced_tab, text='Advanced Search')
        control_tabs.add(view_tab,     text='View')
        control_tabs.add(ml_tab,       text='ML')

        self._build_simple_search_controls(simple_tab)
        self._build_search_bar(advanced_tab)
        self._build_view_options_bar(view_tab)
        # Build ML controls inside its dedicated tab
        self.setup_ml_control_panel(parent=ml_tab)

        # Bind tab-change so the container height follows the active tab
        try:
            control_tabs.bind('<<NotebookTabChanged>>', lambda e: self._on_control_tab_changed(e))
        except Exception:
            pass

        # Main container with left panel, content area, and right sidebar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_panel = ttk.Frame(main_container, width=220)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left_panel.pack_propagate(False)

        ttk.Label(left_panel, text="Folders", font=('Arial', 11, 'bold')).pack(pady=(4, 2))

        nav_frame = ttk.Frame(left_panel)
        nav_frame.pack(fill=tk.BOTH, expand=True)

        self.nav_tree = ttk.Treeview(nav_frame, show='tree', selectmode='browse')
        _nav_sb = ttk.Scrollbar(nav_frame, orient='vertical', command=self.nav_tree.yview)
        self.nav_tree.configure(yscrollcommand=_nav_sb.set)
        _nav_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.nav_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.nav_tree.bind('<<TreeviewOpen>>', self._on_nav_tree_open)
        self.nav_tree.bind('<<TreeviewSelect>>', self._on_nav_tree_select)

        self._populate_nav_tree()
        
        # Center panel - content area
        self.content_frame = ttk.Frame(main_container)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create both views
        self.create_detailed_view()
        self.create_icons_view()
        
        # Show detailed view by default
        self.show_detailed_view()
        
        # Right sidebar - Tag Management
        self.setup_tag_sidebar(main_container)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - ML system initializing...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # History for back button
        self.history = []

        # --- Clipboard state (Cut / Copy / Paste) ---
        self._clipboard_paths: list = []
        self._clipboard_op: str | None = None   # 'copy' or 'cut'

        # --- Internal drag-and-drop state ---
        self._drag_active: bool = False
        self._drag_start_pos: tuple = (0, 0)
        self._drag_source_items: list = []       # list[tuple[iid, Path]]
        self._drag_highlight_id: str | None = None
        # Drag source path used by the icon-view drag handler
        self._drag_source_path: Path | None = None

        # Register OS drag-and-drop targets once widgets are fully realised
        self.root.after(200, self._setup_os_drop_target)
    
    def setup_ml_control_panel(self, parent=None):
        """Setup comprehensive ML control panel with standard and advanced sections

        If `parent` is provided, the panel will be packed into that parent (useful
        for placing the controls inside a Notebook tab). Otherwise it falls back
        to `self.root` for backward compatibility.
        """
        if parent is None:
            parent = self.root
        # Configure font style (minimum 12pt)
        normal_font = ('Arial', 12)
        bold_font = ('Arial', 12, 'bold')
        advanced_font = ('Arial', 11)
        
        # Outer container — packs at the top and sizes itself purely to content.
        # Sections are stacked TOP→BOTTOM (vertical accordion) so collapsing one
        # actually reduces the panel height.  No expand=True anywhere.
        container = ttk.Frame(parent)
        container.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        panel = ttk.Frame(container, relief='groove', borderwidth=1)
        panel.pack(side=tk.TOP, fill=tk.X)

        # Shared vars for ML settings shown in multiple sub-panels
        self.auto_load_var = tk.BooleanVar(value=config_manager.app_config.auto_load_model)
        self.trained_model_var = tk.StringVar(value='latest')

        # ===== TOP: STANDARD SECTION (independently toggleable) =====
        standard_outer = ttk.Frame(panel)
        standard_outer.pack(side=tk.TOP, fill=tk.X)

        # Header row — spans full width, acts as the collapse toggle
        standard_header = ttk.Frame(standard_outer)
        standard_header.pack(fill=tk.X)
        self.standard_expanded = tk.BooleanVar(value=True)
        self.standard_toggle_btn = ttk.Button(standard_header, text="▼ Standard",
                                              command=self.toggle_standard_section)
        self.standard_toggle_btn.pack(side=tk.LEFT, pady=1, padx=1)

        # Content frame
        self.standard_content = ttk.Frame(standard_outer)
        self.standard_content.pack(fill=tk.X)

        # 2-column grid inside Standard content
        standard_grid = ttk.Frame(self.standard_content)
        standard_grid.pack(fill=tk.X, padx=1, pady=1)
        
        # Left column: Model & Status
        col_left = ttk.Frame(standard_grid)
        col_left.grid(row=0, column=0, sticky='nsew', padx=(0, 1))
        
        ttk.Label(col_left, text="Model:", font=bold_font).pack(anchor='w')
        self.model_type_var = tk.StringVar(value=config_manager.app_config.model_type)
        model_options = ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b1', 'mobilenet_v2', 'vgg16']
        self.model_dropdown = ttk.Combobox(col_left, textvariable=self.model_type_var, 
                                          values=model_options, state='readonly', width=18, font=normal_font)
        self.model_dropdown.pack(fill=tk.X, pady=1)
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        ttk.Label(col_left, text="Status:", font=bold_font).pack(anchor='w', pady=(1, 0))
        self.model_status_label = ttk.Label(col_left, text="No model", foreground='gray', 
                                           wraplength=280, justify='left', font=normal_font)
        self.model_status_label.pack(anchor='w')
        
        ttk.Label(col_left, text="Mode:", font=bold_font).pack(anchor='w', pady=(1, 0))
        self.training_mode_var = tk.StringVar(value='supervised' if config_manager.app_config.supervised_mode else 'unsupervised')
        mode_options = ['supervised', 'unsupervised']
        self.mode_dropdown = ttk.Combobox(col_left, textvariable=self.training_mode_var,
                                         values=mode_options, state='readonly', width=18, font=normal_font)
        self.mode_dropdown.pack(fill=tk.X, pady=1)
        self.mode_dropdown.bind('<<ComboboxSelected>>', self.on_mode_change)

        # Training Source
        source_lf = ttk.LabelFrame(col_left, text="Training Source", padding=2)
        source_lf.pack(fill=tk.X, pady=(4, 1))
        self._train_source_var = tk.StringVar(value='folders')
        ttk.Radiobutton(source_lf, text="Folders only", variable=self._train_source_var,
                        value='folders', command=self._on_train_source_change).pack(anchor='w')
        ttk.Radiobutton(source_lf, text="COCO only", variable=self._train_source_var,
                        value='coco', command=self._on_train_source_change).pack(anchor='w')
        ttk.Radiobutton(source_lf, text="Folders + COCO", variable=self._train_source_var,
                        value='both', command=self._on_train_source_change).pack(anchor='w')

        self._train_coco_settings = ttk.Frame(col_left)
        coco_dir_row = ttk.Frame(self._train_coco_settings)
        coco_dir_row.pack(fill=tk.X)
        ttk.Label(coco_dir_row, text="COCO dir:", font=normal_font).pack(side=tk.LEFT)
        self._train_coco_dir_var = tk.StringVar(
            value=config_manager.app_config.coco_data_dir or './data/coco')
        ttk.Entry(coco_dir_row, textvariable=self._train_coco_dir_var,
                  font=normal_font).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(coco_dir_row, text="…", width=3,
                   command=self._browse_train_coco_dir).pack(side=tk.LEFT)
        # Hidden until COCO is selected

        # Trained-model selection + auto-load toggle
        ttk.Label(col_left, text="Trained:", font=bold_font).pack(anchor='w', pady=(6, 0))
        trained_row = ttk.Frame(col_left)
        trained_row.pack(fill=tk.X, pady=1)
        self.trained_model_dropdown = ttk.Combobox(
            trained_row,
            textvariable=self.trained_model_var,
            values=[],
            state='readonly',
            width=18,
            font=normal_font,
        )
        self.trained_model_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Refresh list just-in-time before opening
        self.trained_model_dropdown.bind('<Button-1>', lambda e: self._refresh_trained_models_async())

        ttk.Checkbutton(
            col_left,
            text="Auto-load model",
            variable=self.auto_load_var,
            command=self._on_auto_load_toggle,
        ).pack(anchor='w', pady=(4, 1))
        
        # Right column: Actions
        col_right = ttk.Frame(standard_grid)
        col_right.grid(row=0, column=1, sticky='nsew', padx=(1, 0))
        
        ttk.Label(col_right, text="Actions:", font=bold_font).pack(anchor='w')
        
        self.train_folders_btn = ttk.Button(col_right, text="Train",
                                           command=self.train_from_source)
        self.train_folders_btn.pack(fill=tk.X, pady=1)
        
        self.train_tags_btn = ttk.Button(col_right, text="🏷️ Train Tags",
                                        command=self.train_from_tags)
        self.train_tags_btn.pack(fill=tk.X, pady=1)
        
        self.auto_sort_btn = ttk.Button(col_right, text="🎯 Auto-Sort",
                                       command=self.auto_sort_images)
        self.auto_sort_btn.pack(fill=tk.X, pady=1)
        
        self.cluster_btn = ttk.Button(col_right, text="🔮 Clustering",
                                     command=self.run_clustering)
        self.cluster_btn.pack(fill=tk.X, pady=1)

        # Manual model load (user-triggered)
        self.load_model_btn = ttk.Button(col_right, text="⬇️ Load Model",
                         command=self.on_load_model)
        self.load_model_btn.pack(fill=tk.X, pady=(6, 1))
        
        # Configure grid weights
        standard_grid.grid_columnconfigure(0, weight=1)
        standard_grid.grid_columnconfigure(1, weight=1)
        
        # ===== BOTTOM: ADVANCED SECTION (independently toggleable) =====
        # Packed below Standard so collapsing either section changes total height.
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X)
        advanced_outer = ttk.Frame(panel)
        advanced_outer.pack(side=tk.TOP, fill=tk.X)

        # Advanced section header with its own toggle button
        advanced_header = ttk.Frame(advanced_outer)
        advanced_header.pack(fill=tk.X)
        self.advanced_expanded = tk.BooleanVar(value=True)
        self.advanced_toggle_btn = ttk.Button(advanced_header, text="▼ Advanced",
                                              command=self.toggle_advanced_section)
        self.advanced_toggle_btn.pack(side=tk.LEFT, pady=1, padx=1)

        # Advanced content frame
        self.advanced_content = ttk.Frame(advanced_outer)
        self.advanced_content.pack(fill=tk.X)   # fill=X, not BOTH

        # ── Profile bar ──────────────────────────────────────────────────
        from config import profile_manager as _pm
        profile_frame = ttk.LabelFrame(self.advanced_content, text="Profiles", padding=2)
        profile_frame.pack(fill=tk.X, padx=1, pady=(2, 2))
        profile_row = ttk.Frame(profile_frame)
        profile_row.pack(fill=tk.X)
        self._profile_var = tk.StringVar()
        self._profile_combo = ttk.Combobox(
            profile_row, textvariable=self._profile_var,
            values=_pm.list_names(), state='readonly',
            width=26, font=advanced_font)
        self._profile_combo.pack(side=tk.LEFT, padx=(0, 4))
        if _pm.list_names():
            self._profile_combo.set(_pm.list_names()[0])
        ttk.Button(profile_row, text="Load",    width=6,
                   command=self._load_profile).pack(side=tk.LEFT, padx=2)
        ttk.Button(profile_row, text="Save As", width=8,
                   command=self._save_profile_as).pack(side=tk.LEFT, padx=2)
        ttk.Button(profile_row, text="Delete",  width=6,
                   command=self._delete_profile).pack(side=tk.LEFT, padx=2)

        # 3-column grid for hyperparameters — no scrollbar needed
        params_grid = ttk.Frame(self.advanced_content)
        params_grid.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        self.hyper_entries = {}
        
        # Column 1: Training Parameters
        col1 = ttk.LabelFrame(params_grid, text="Training", padding=1)
        col1.grid(row=0, column=0, sticky='nsew', padx=1, pady=1)
        
        r = 0
        ttk.Label(col1, text="LR:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['learning_rate'] = ttk.Entry(col1, width=10, font=advanced_font)
        self.hyper_entries['learning_rate'].insert(0, str(config_manager.hyperparameters.learning_rate))
        self.hyper_entries['learning_rate'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        r += 1
        ttk.Label(col1, text="Batch:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['batch_size'] = ttk.Entry(col1, width=10, font=advanced_font)
        self.hyper_entries['batch_size'].insert(0, str(config_manager.hyperparameters.batch_size))
        self.hyper_entries['batch_size'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        r += 1
        ttk.Label(col1, text="Epochs:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['epochs'] = ttk.Entry(col1, width=10, font=advanced_font)
        self.hyper_entries['epochs'].insert(0, str(config_manager.hyperparameters.epochs))
        self.hyper_entries['epochs'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        r += 1
        ttk.Label(col1, text="Dropout:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['dropout_rate'] = ttk.Entry(col1, width=10, font=advanced_font)
        self.hyper_entries['dropout_rate'].insert(0, str(config_manager.hyperparameters.dropout_rate))
        self.hyper_entries['dropout_rate'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        r += 1
        ttk.Label(col1, text="Patience:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['early_stopping_patience'] = ttk.Entry(col1, width=10, font=advanced_font)
        self.hyper_entries['early_stopping_patience'].insert(0, str(config_manager.hyperparameters.early_stopping_patience))
        self.hyper_entries['early_stopping_patience'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        col1.grid_columnconfigure(1, weight=1)
        
        # Column 2: Data & Validation
        col2 = ttk.LabelFrame(params_grid, text="Data", padding=1)
        col2.grid(row=0, column=1, sticky='nsew', padx=1, pady=1)
        
        r = 0
        ttk.Label(col2, text="Val Split:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['validation_split'] = ttk.Entry(col2, width=10, font=advanced_font)
        self.hyper_entries['validation_split'].insert(0, str(config_manager.hyperparameters.validation_split))
        self.hyper_entries['validation_split'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        r += 1
        ttk.Label(col2, text="Conf Thr:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['confidence_threshold'] = ttk.Entry(col2, width=10, font=advanced_font)
        self.hyper_entries['confidence_threshold'].insert(0, str(config_manager.hyperparameters.confidence_threshold))
        self.hyper_entries['confidence_threshold'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        r += 1
        ttk.Label(col2, text="Max Sug:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        self.hyper_entries['max_suggestions'] = ttk.Entry(col2, width=10, font=advanced_font)
        self.hyper_entries['max_suggestions'].insert(0, str(config_manager.hyperparameters.max_suggestions))
        self.hyper_entries['max_suggestions'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        r += 1
        ttk.Label(col2, text="Img Size:", font=advanced_font).grid(row=r, column=0, sticky='w', pady=1, padx=1)
        img_size_str = f"{config_manager.hyperparameters.image_size[0]}"
        self.hyper_entries['image_size'] = ttk.Entry(col2, width=10, font=advanced_font)
        self.hyper_entries['image_size'].insert(0, img_size_str)
        self.hyper_entries['image_size'].grid(row=r, column=1, sticky='ew', padx=1, pady=1)
        
        col2.grid_columnconfigure(1, weight=1)
        
        # Column 3: Advanced Options
        col3 = ttk.LabelFrame(params_grid, text="Options", padding=1)
        col3.grid(row=0, column=2, sticky='nsew', padx=1, pady=1)
        
        self.use_gpu_var = tk.BooleanVar(value=config_manager.app_config.use_gpu)
        ttk.Checkbutton(col3, text="GPU", variable=self.use_gpu_var).pack(anchor='w', pady=1)
        
        self.pretrained_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Pretrained", variable=self.pretrained_var).pack(anchor='w', pady=1)
        
        self.data_aug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Data Aug", variable=self.data_aug_var).pack(anchor='w', pady=1)
        
        ttk.Checkbutton(
            col3,
            text="Auto-load",
            variable=self.auto_load_var,
            command=self._on_auto_load_toggle,
        ).pack(anchor='w', pady=1)

        ttk.Separator(col3, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        self._coco_all_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="All COCO images", variable=self._coco_all_images_var,
                        command=self._on_coco_all_toggle).pack(anchor='w', pady=1)
        self._coco_max_frame = ttk.Frame(col3)
        ttk.Label(self._coco_max_frame, text="Max imgs:", font=advanced_font).pack(side=tk.LEFT)
        self._coco_max_var = tk.IntVar(value=config_manager.app_config.coco_max_images or 500)
        ttk.Spinbox(self._coco_max_frame, from_=50, to=5000, increment=50,
                    textvariable=self._coco_max_var, width=7,
                    font=advanced_font).pack(side=tk.LEFT, padx=2)
        # Hidden until "All COCO images" is unchecked

        worker_frame = ttk.Frame(col3)
        worker_frame.pack(fill=tk.X, pady=1)
        ttk.Label(worker_frame, text="Workers:", font=advanced_font).pack(side=tk.LEFT)
        self.hyper_entries['num_workers'] = ttk.Entry(worker_frame, width=6, font=advanced_font)
        self.hyper_entries['num_workers'].insert(0, str(config_manager.app_config.num_workers))
        self.hyper_entries['num_workers'].pack(side=tk.LEFT, padx=1)
        
        # Action Buttons
        button_frame = ttk.Frame(col3)
        button_frame.pack(fill=tk.X, pady=1)
        
        ttk.Button(button_frame, text="✓", command=self.apply_hyperparameters, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(button_frame, text="↻", command=self.reset_hyperparameters, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(button_frame, text="💾", command=self.save_config_file, width=4).pack(side=tk.LEFT, padx=1)
        
        # Configure grid column weights for equal sizing
        params_grid.grid_columnconfigure(0, weight=1)
        params_grid.grid_columnconfigure(1, weight=1)
        params_grid.grid_columnconfigure(2, weight=1)

        # Initial population (may update later once root folder is set)
        try:
            self._refresh_trained_models_async()
        except Exception:
            pass

    def _on_train_source_change(self):
        if self._train_source_var.get() in ('coco', 'both'):
            self._train_coco_settings.pack(fill=tk.X, pady=(2, 0))
        else:
            self._train_coco_settings.pack_forget()

    def _browse_train_coco_dir(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(
            title="Select COCO data directory",
            initialdir=self._train_coco_dir_var.get())
        if folder:
            self._train_coco_dir_var.set(folder)

    def _on_coco_all_toggle(self):
        if self._coco_all_images_var.get():
            self._coco_max_frame.pack_forget()
        else:
            self._coco_max_frame.pack(fill=tk.X, pady=1)

    def train_from_source(self):
        source = self._train_source_var.get()
        label = {'folders': 'Folders', 'coco': 'COCO', 'both': 'Folders + COCO'}[source]
        if not messagebox.askyesno("Confirm",
                f"This will train a new model using {label} as the data source. Continue?"):
            return

        coco_dir = self._train_coco_dir_var.get().strip() or './data/coco'
        use_all = self._coco_all_images_var.get()
        coco_max = 0 if use_all else int(self._coco_max_var.get())

        if source in ('coco', 'both'):
            config_manager.app_config.coco_data_dir = coco_dir
            if not use_all:
                config_manager.app_config.coco_max_images = coco_max
            config_manager.save_config()

        state = _TrainingState()
        progress_dialog = TrainingProgressDialog(
            self.root, f"Training from {label}",
            cancel_callback=lambda: self.controller.training_manager.cancel())
        progress_dialog.show()

        def on_complete():
            self.status_var.set("Training completed successfully")
            self.model_status_label.config(
                text=f"✓ Model trained & loaded ({config_manager.app_config.model_type})",
                foreground='green')
            try:
                self._refresh_trained_models_async()
            except Exception:
                pass

        self.controller.training_manager.progress_callback = state.push
        state.push('status', 'Initializing training...')

        def _run():
            try:
                if source == 'folders':
                    self.controller.bootstrap_from_folders()
                elif source == 'coco':
                    self.controller.bootstrap_from_coco(coco_dir=coco_dir, coco_max=coco_max)
                else:
                    self.controller.bootstrap_from_both(coco_dir=coco_dir, coco_max=coco_max)
            except Exception as e:
                state.push('error', str(e))

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(150, self._make_training_poll(state, progress_dialog, on_complete))

    def on_model_change(self, event=None):
        """Handle model selection change"""
        new_model = self.model_type_var.get()
        config_manager.app_config.model_type = new_model
        print(f"Model changed to: {new_model}")
        self.model_status_label.config(text=f"Model changed to {new_model} (not loaded)", foreground='orange')
    
    def on_mode_change(self, event=None):
        """Handle training mode change"""
        new_mode = self.training_mode_var.get()
        config_manager.app_config.supervised_mode = (new_mode == 'supervised')
        print(f"Training mode changed to: {new_mode}")
        
        # Update button states based on mode
        if new_mode == 'supervised':
            self.train_folders_btn.config(state=tk.NORMAL)
            self.train_tags_btn.config(state=tk.NORMAL)
            self.auto_sort_btn.config(state=tk.NORMAL)
            self.cluster_btn.config(state=tk.DISABLED)
        else:
            self.train_folders_btn.config(state=tk.DISABLED)
            self.train_tags_btn.config(state=tk.DISABLED)
            self.auto_sort_btn.config(state=tk.DISABLED)
            self.cluster_btn.config(state=tk.NORMAL)
    
    def toggle_standard_section(self):
        """Toggle the Standard sub-section inside ML Controls."""
        if self.standard_expanded.get():
            self.standard_content.pack_forget()
            self.standard_toggle_btn.config(text="▶ Standard")
            self.standard_expanded.set(False)
        else:
            self.standard_content.pack(fill=tk.X)   # content-driven height
            self.standard_toggle_btn.config(text="▼ Standard")
            self.standard_expanded.set(True)

    def toggle_advanced_section(self):
        """Toggle the Advanced sub-section inside ML Controls."""
        if self.advanced_expanded.get():
            self.advanced_content.pack_forget()
            self.advanced_toggle_btn.config(text="▶ Advanced")
            self.advanced_expanded.set(False)
        else:
            self.advanced_content.pack(fill=tk.X)   # content-driven height
            self.advanced_toggle_btn.config(text="▼ Advanced")
            self.advanced_expanded.set(True)
    
    def apply_hyperparameters(self):
        """Apply hyperparameter changes from UI to config"""
        try:
            # Update hyperparameters
            config_manager.hyperparameters.learning_rate = float(self.hyper_entries['learning_rate'].get())
            config_manager.hyperparameters.batch_size = int(self.hyper_entries['batch_size'].get())
            config_manager.hyperparameters.epochs = int(self.hyper_entries['epochs'].get())
            config_manager.hyperparameters.dropout_rate = float(self.hyper_entries['dropout_rate'].get())
            config_manager.hyperparameters.early_stopping_patience = int(self.hyper_entries['early_stopping_patience'].get())
            config_manager.hyperparameters.validation_split = float(self.hyper_entries['validation_split'].get())
            config_manager.hyperparameters.confidence_threshold = float(self.hyper_entries['confidence_threshold'].get())
            config_manager.hyperparameters.max_suggestions = int(self.hyper_entries['max_suggestions'].get())
            
            # Image size
            img_size = int(self.hyper_entries['image_size'].get())
            config_manager.hyperparameters.image_size = (img_size, img_size)
            
            # App config
            config_manager.app_config.use_gpu = self.use_gpu_var.get()
            config_manager.app_config.num_workers = int(self.hyper_entries['num_workers'].get())
            # Auto-load preference
            config_manager.app_config.auto_load_model = bool(self.auto_load_var.get())
            
            messagebox.showinfo("Success", "Hyperparameters applied successfully!\nThese will be used for the next training session.")
            print("✓ Hyperparameters updated")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {e}")

    def _on_auto_load_toggle(self) -> None:
        """Persist the Auto-load toggle immediately."""
        try:
            config_manager.app_config.auto_load_model = bool(self.auto_load_var.get())
            config_manager.save_config()
        except Exception:
            pass

    def _refresh_trained_models(self) -> None:
        """Refresh the trained-model dropdown from disk."""
        try:
            from data_layer import model_storage
            models = model_storage.list_models()
        except Exception:
            models = []

        # Ensure legacy 'latest' shows up if present.
        try:
            from data_layer import model_storage
            if model_storage.model_exists('latest') and 'latest' not in models:
                models = ['latest'] + models
        except Exception:
            pass

        # Update combobox values
        try:
            self.trained_model_dropdown.configure(values=models)
        except Exception:
            return

        # Keep current selection if possible, otherwise pick a sane default
        current = str(self.trained_model_var.get() or '').strip()
        if current and current in models:
            return
        if 'latest' in models:
            self.trained_model_var.set('latest')
        elif models:
            self.trained_model_var.set(models[0])
        else:
            self.trained_model_var.set('')

    def _refresh_trained_models_async(self) -> None:
        """Non-blocking wrapper around `_refresh_trained_models()`.

        Scanning the model directory can block on slow disks or AV scanners.
        This keeps the UI responsive by doing the scan in a worker thread.
        """
        try:
            if getattr(self, '_trained_models_refresh_inflight', False):
                return
            self._trained_models_refresh_inflight = True
        except Exception:
            return

        def _worker() -> None:
            t0 = time.time()
            try:
                from data_layer import model_storage
                models = model_storage.list_models()
                try:
                    if model_storage.model_exists('latest') and 'latest' not in models:
                        models = ['latest'] + models
                except Exception:
                    pass
            except Exception:
                models = []
            dt = time.time() - t0

            def _apply() -> None:
                try:
                    self._trained_models_refresh_inflight = False
                except Exception:
                    pass
                try:
                    if not hasattr(self, 'trained_model_dropdown'):
                        return
                    if not self.root.winfo_exists():
                        return
                except Exception:
                    return

                try:
                    self.trained_model_dropdown.configure(values=models)
                except Exception:
                    return

                current = str(self.trained_model_var.get() or '').strip()
                if current and current in models:
                    return
                if 'latest' in models:
                    self.trained_model_var.set('latest')
                elif models:
                    self.trained_model_var.set(models[0])
                else:
                    self.trained_model_var.set('')

                if dt > 0.25:
                    try:
                        print(f"[ML] model list refresh took {dt:.3f}s ({len(models)} model(s))")
                    except Exception:
                        pass

            try:
                self.root.after(0, _apply)
            except Exception:
                try:
                    self._trained_models_refresh_inflight = False
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()
    
    def reset_hyperparameters(self):
        """Reset hyperparameters to defaults"""
        if messagebox.askyesno("Confirm Reset", "Reset all hyperparameters to default values?"):
            config_manager.hyperparameters = HyperParameters()
            config_manager.app_config.use_gpu = True
            config_manager.app_config.num_workers = 4
            
            # Update UI
            self.hyper_entries['learning_rate'].delete(0, tk.END)
            self.hyper_entries['learning_rate'].insert(0, str(config_manager.hyperparameters.learning_rate))
            self.hyper_entries['batch_size'].delete(0, tk.END)
            self.hyper_entries['batch_size'].insert(0, str(config_manager.hyperparameters.batch_size))
            self.hyper_entries['epochs'].delete(0, tk.END)
            self.hyper_entries['epochs'].insert(0, str(config_manager.hyperparameters.epochs))
            self.hyper_entries['dropout_rate'].delete(0, tk.END)
            self.hyper_entries['dropout_rate'].insert(0, str(config_manager.hyperparameters.dropout_rate))
            self.hyper_entries['early_stopping_patience'].delete(0, tk.END)
            self.hyper_entries['early_stopping_patience'].insert(0, str(config_manager.hyperparameters.early_stopping_patience))
            self.hyper_entries['validation_split'].delete(0, tk.END)
            self.hyper_entries['validation_split'].insert(0, str(config_manager.hyperparameters.validation_split))
            self.hyper_entries['confidence_threshold'].delete(0, tk.END)
            self.hyper_entries['confidence_threshold'].insert(0, str(config_manager.hyperparameters.confidence_threshold))
            self.hyper_entries['max_suggestions'].delete(0, tk.END)
            self.hyper_entries['max_suggestions'].insert(0, str(config_manager.hyperparameters.max_suggestions))
            self.hyper_entries['image_size'].delete(0, tk.END)
            self.hyper_entries['image_size'].insert(0, str(config_manager.hyperparameters.image_size[0]))
            self.hyper_entries['num_workers'].delete(0, tk.END)
            self.hyper_entries['num_workers'].insert(0, str(config_manager.app_config.num_workers))
            
            self.use_gpu_var.set(True)
            self.pretrained_var.set(True)
            self.data_aug_var.set(True)
            self.auto_load_var.set(False)
            try:
                config_manager.app_config.auto_load_model = False
            except Exception:
                pass
            
            messagebox.showinfo("Success", "Hyperparameters reset to defaults")
    
    def save_config_file(self):
        """Save current configuration to file"""
        try:
            # First apply current UI values
            self.apply_hyperparameters()
            # Then save to file
            config_manager.save_config()
            messagebox.showinfo("Success", f"Configuration saved to {config_manager.config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    # ------------------------------------------------------------------
    # Training profile methods
    # ------------------------------------------------------------------

    def _refresh_profile_list(self) -> None:
        from config import profile_manager as _pm
        self._profile_combo['values'] = _pm.list_names()

    def _load_profile(self) -> None:
        from config import profile_manager as _pm
        name = self._profile_var.get()
        p = _pm.get(name)
        if p is None:
            return
        field_values = {
            'learning_rate':           str(p.learning_rate),
            'batch_size':              str(p.batch_size),
            'epochs':                  str(p.epochs),
            'dropout_rate':            str(p.dropout_rate),
            'early_stopping_patience': str(p.early_stopping_patience),
            'validation_split':        str(p.validation_split),
            'confidence_threshold':    str(p.confidence_threshold),
            'max_suggestions':         str(p.max_suggestions),
            'image_size':              str(p.image_size),
        }
        for key, val in field_values.items():
            entry = self.hyper_entries.get(key)
            if entry is not None:
                entry.delete(0, tk.END)
                entry.insert(0, val)
        self.model_type_var.set(p.model_type)
        self.status_var.set(f"Profile '{name}' loaded — click ✓ to apply.")

    def _save_profile_as(self) -> None:
        from config import profile_manager as _pm, TrainingProfile
        name = simpledialog.askstring('Save Profile', 'Profile name:', parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        try:
            p = TrainingProfile(
                name=name,
                model_type=self.model_type_var.get(),
                learning_rate=float(self.hyper_entries['learning_rate'].get()),
                batch_size=int(self.hyper_entries['batch_size'].get()),
                epochs=int(self.hyper_entries['epochs'].get()),
                dropout_rate=float(self.hyper_entries['dropout_rate'].get()),
                early_stopping_patience=int(self.hyper_entries['early_stopping_patience'].get()),
                validation_split=float(self.hyper_entries['validation_split'].get()),
                confidence_threshold=float(self.hyper_entries['confidence_threshold'].get()),
                max_suggestions=int(self.hyper_entries['max_suggestions'].get()),
                image_size=int(self.hyper_entries['image_size'].get()),
            )
            _pm.save(p)
            self._refresh_profile_list()
            self._profile_var.set(name)
            self.status_var.set(f"Profile '{name}' saved.")
        except ValueError as e:
            messagebox.showerror('Save Profile', f'Invalid value: {e}')

    def _delete_profile(self) -> None:
        from config import profile_manager as _pm
        name = self._profile_var.get()
        if not name:
            return
        if _pm.is_builtin(name):
            messagebox.showwarning('Delete Profile',
                                   f"'{name}' is a built-in profile and cannot be deleted.")
            return
        if not messagebox.askyesno('Delete Profile', f"Delete profile '{name}'?"):
            return
        _pm.delete(name)
        self._refresh_profile_list()
        names = _pm.list_names()
        self._profile_var.set(names[0] if names else '')
        self.status_var.set(f"Profile '{name}' deleted.")

    def _populate_nav_tree(self) -> None:
        self.nav_tree.delete(*self.nav_tree.get_children())
        if platform.system() == 'Windows':
            drives: list[str] = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
                    # Insert immediately with bare label; free-space label is filled in async
                    iid = self.nav_tree.insert('', 'end', iid=drive, text=drive,
                                               values=(drive,), open=False)
                    self.nav_tree.insert(iid, 'end', text='_loading_')

            def _fetch_labels() -> None:
                labels: dict[str, str] = {}
                for d in drives:
                    try:
                        _, _, free = self.get_drive_space(d)
                        labels[d] = f"{d}  ({self.format_size(free)} free)"
                    except Exception:
                        labels[d] = d

                def _apply() -> None:
                    for d, lbl in labels.items():
                        try:
                            self.nav_tree.item(d, text=lbl)
                        except Exception:
                            pass
                try:
                    self.root.after(0, _apply)
                except Exception:
                    pass

            threading.Thread(target=_fetch_labels, daemon=True).start()
        else:
            for root_path in ('/', '/home', '/media', '/mnt', '/Volumes'):
                if os.path.exists(root_path):
                    iid = self.nav_tree.insert('', 'end', iid=root_path,
                                               text=root_path, values=(root_path,), open=False)
                    self.nav_tree.insert(iid, 'end', text='_loading_')

    def _on_nav_tree_open(self, *_) -> None:
        item = self.nav_tree.focus()
        children = self.nav_tree.get_children(item)
        if len(children) == 1 and self.nav_tree.item(children[0], 'text') == '_loading_':
            self.nav_tree.delete(children[0])
            vals = self.nav_tree.item(item, 'values')
            path_str = vals[0] if vals else self.nav_tree.item(item, 'text')

            def _scan() -> None:
                entries: list[tuple[str, str]] = []
                try:
                    path = Path(path_str)
                    for entry in sorted(os.scandir(path), key=lambda e: e.name.lower()):
                        if (entry.is_dir(follow_symlinks=False)
                                and not entry.name.startswith('.')
                                and entry.name.lower() != _THUMBS_FOLDER):
                            try:
                                if not self._is_os_hidden_entry(entry):
                                    entries.append((entry.name, entry.path))
                            except PermissionError:
                                pass
                except Exception:
                    pass

                def _apply() -> None:
                    try:
                        if not self.nav_tree.exists(item):
                            return
                    except Exception:
                        return
                    for name, path_s in entries:
                        try:
                            ciid = self.nav_tree.insert(
                                item, 'end', text=name,
                                values=(path_s,), open=False)
                            self.nav_tree.insert(ciid, 'end', text='_loading_')
                        except Exception:
                            pass

                try:
                    self.root.after(0, _apply)
                except Exception:
                    pass

            threading.Thread(target=_scan, daemon=True).start()

    def _on_nav_tree_select(self, *_) -> None:
        if getattr(self, '_nav_tree_revealing', False):
            return   # selection set programmatically — don't navigate
        item = self.nav_tree.focus()
        if not item:
            return
        vals = self.nav_tree.item(item, 'values')
        path_str = vals[0] if vals else self.nav_tree.item(item, 'text')
        try:
            path = Path(path_str)
            if path.is_dir() and path != self.current_path:
                self.history.append(self.current_path)
                self._nav_forward_stack.clear()
                self.load_directory(path)
        except Exception:
            pass

    def _nav_tree_expand_sync(self, iid: str) -> None:
        """Synchronously expand one nav-tree node (used during path reveal)."""
        children = self.nav_tree.get_children(iid)
        if len(children) == 1 and self.nav_tree.item(children[0], 'text') == '_loading_':
            self.nav_tree.delete(children[0])
            vals = self.nav_tree.item(iid, 'values')
            path_str = vals[0] if vals else self.nav_tree.item(iid, 'text')
            try:
                for entry in sorted(os.scandir(path_str), key=lambda e: e.name.lower()):
                    if (entry.is_dir(follow_symlinks=False)
                            and not entry.name.startswith('.')
                            and entry.name.lower() != _THUMBS_FOLDER):
                        try:
                            if not self._is_os_hidden_entry(entry):
                                ciid = self.nav_tree.insert(
                                    iid, 'end', text=entry.name,
                                    values=(entry.path,), open=False)
                                self.nav_tree.insert(ciid, 'end', text='_loading_')
                        except PermissionError:
                            pass
            except Exception:
                pass
        try:
            self.nav_tree.item(iid, open=True)
        except Exception:
            pass

    def _nav_tree_reveal_path(self, target: Path) -> None:
        """Expand the side nav tree to *target* and select it.

        Only walks the path components that correspond to *target*, so the
        synchronous scandir calls are limited to the depth of the target path
        rather than the whole drive.
        """
        try:
            target = target.resolve()
            parts = list(target.parts)   # e.g. ['D:\\', 'Projects', 'App']
            if not parts:
                return

            root_iid = parts[0]          # 'D:\\' on Windows, '/' on Unix
            if not self.nav_tree.exists(root_iid):
                return

            self._nav_tree_expand_sync(root_iid)
            current_iid = root_iid

            for depth, part in enumerate(parts[1:], start=1):
                partial = str(Path(*parts[:depth + 1]))
                found: str | None = None
                for child_iid in self.nav_tree.get_children(current_iid):
                    child_vals = self.nav_tree.item(child_iid, 'values')
                    if child_vals and Path(child_vals[0]) == Path(partial):
                        found = child_iid
                        break
                if found is None:
                    break
                # Only expand intermediate nodes (don't expand the leaf)
                if depth < len(parts) - 1:
                    self._nav_tree_expand_sync(found)
                current_iid = found

            try:
                self._nav_tree_revealing = True
                self.nav_tree.selection_set(current_iid)
                self.nav_tree.see(current_iid)
            except Exception:
                pass
            finally:
                self._nav_tree_revealing = False
        except Exception:
            self._nav_tree_revealing = False

    def get_drive_space(self, drive):
        """Get drive space information"""
        if platform.system() == 'Windows':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            used_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(drive),
                ctypes.pointer(free_bytes),
                ctypes.pointer(total_bytes),
                None
            )
            return total_bytes.value, used_bytes.value, free_bytes.value
        else:
            # Unix/Linux - use statvfs
            try:
                stat = os.statvfs(drive)  # type: ignore
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_bavail * stat.f_frsize
                used = total - free
                return total, used, free
            except AttributeError:
                # Fallback if statvfs not available
                return 0, 0, 0
    
    def create_detailed_view(self):
        """Create the detailed list view"""
        self.detailed_frame = ttk.Frame(self.content_frame)
        
        # Create Treeview with additional columns
        columns = ('Type', 'Size', 'Date Created', 'Date Modified')
        self.tree = ttk.Treeview(self.detailed_frame, columns=columns, show='tree headings')
        
        # Configure columns
        self.tree.heading('#0', text='Name', anchor=tk.W)
        self.tree.heading('Type', text='Type', anchor=tk.W)
        self.tree.heading('Size', text='Size', anchor=tk.E)
        self.tree.heading('Date Created', text='Date Created', anchor=tk.W)
        self.tree.heading('Date Modified', text='Date Modified', anchor=tk.W)
        
        self.tree.column('#0', width=300)
        self.tree.column('Type', width=100)
        self.tree.column('Size', width=100)
        self.tree.column('Date Created', width=180)
        self.tree.column('Date Modified', width=180)
        
        # Scrollbars
        vsb = ttk.Scrollbar(self.detailed_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.detailed_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        self.detailed_frame.grid_rowconfigure(0, weight=1)
        self.detailed_frame.grid_columnconfigure(0, weight=1)
        
        # Bind double-click, single-click, and right-click events
        self.tree.bind('<Double-1>', self.on_tree_double_click)
        self.tree.bind('<ButtonRelease-1>', self.on_tree_single_click)
        self.tree.bind('<Button-3>', self._show_tree_context_menu)

        # Keyboard shortcuts for clipboard operations
        self.tree.bind('<Control-c>', lambda e: self._clip_copy())
        self.tree.bind('<Control-x>', lambda e: self._clip_cut())
        self.tree.bind('<Control-v>', lambda e: self._clip_paste())
        self.tree.bind('<F2>',        lambda e: self._file_rename())
        self.tree.bind('<Delete>',    lambda e: self._file_delete())

        # Internal drag-and-drop
        self.tree.bind('<ButtonPress-1>',   self._on_tree_drag_start)
        self.tree.bind('<B1-Motion>',       self._on_tree_drag_motion)
        self.tree.bind('<ButtonRelease-1>', self._on_tree_drag_release)
    
    def create_icons_view(self):
        """Create the large icons view"""
        self.icons_frame = ttk.Frame(self.content_frame)
        
        # Create canvas with scrollbar
        self.canvas = Canvas(self.icons_frame, bg='white')
        vsb = ttk.Scrollbar(self.icons_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.icons_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.icons_container, anchor='nw')
        
        self.icons_container.bind('<Configure>', self.on_icons_configure)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        
        self.canvas.bind_all('<MouseWheel>', self.on_mousewheel)
        
        self.icon_buttons = []
        
        self.tree_thumbnails = {}
        # Bounded FIFO PhotoImage cache — evicted entries are blanked to free Tk memory
        self.thumbnail_cache: dict = {}
        self._THUMB_CACHE_MAX = 512
        self._thumb_lock = threading.RLock()
        self._thumb_generation = 0

        self._nav_generation = 0

        self._placeholder_cache: dict = {}

        # Gallery layout state
        self._gallery_items: tuple | None = None
        self._gallery_last_thumb_size: int = 0
        self._gallery_relayout_id: str | None = None

        # After-call ID used to debounce rapid single-click tag lookups.
        self._tag_debounce_id: str | None = None
    
    def on_icons_configure(self, event):
        """Update scroll region when icons container changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def on_canvas_configure(self, event):
        """Update canvas window width to match canvas; re-layout gallery on resize."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        if (self.view_mode == 'icons'
                and hasattr(self, '_gallery_items')
                and self._gallery_items is not None):
            if hasattr(self, '_gallery_relayout_id') and self._gallery_relayout_id:
                self.root.after_cancel(self._gallery_relayout_id)
            self._gallery_relayout_id = self.root.after(80, self._relayout_gallery)
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling — skip if the pointer is inside the tag sidebar."""
        # Walk up the widget parent chain to see if the event came from the tag sidebar.
        if hasattr(self, 'tag_sidebar'):
            w = event.widget
            while w is not None:
                if w is self.tag_sidebar:
                    return  # let the tag sidebar handle its own scroll
                try:
                    w = w.nametowidget(w.winfo_parent())
                except Exception:
                    break
        if self.view_mode == 'icons':
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def show_detailed_view(self):
        """Show the detailed list view"""
        self.icons_frame.pack_forget()
        self.detailed_frame.pack(fill=tk.BOTH, expand=True)
        self.view_mode = 'detailed'
        try:
            self._update_view_toggle_buttons()
        except Exception:
            pass
    
    def show_icons_view(self):
        """Show the large icons view"""
        self.detailed_frame.pack_forget()
        self.icons_frame.pack(fill=tk.BOTH, expand=True)
        self.view_mode = 'icons'
        try:
            self._update_view_toggle_buttons()
        except Exception:
            pass
    
    def switch_to_detailed(self):
        """Switch to detailed view"""
        if self.view_mode != 'detailed':
            self.show_detailed_view()
            if getattr(self, '_search_active', False):
                self._display_search_results(getattr(self, '_last_search_results', []))
            else:
                self.load_directory(self.current_path)

    def switch_to_icons(self):
        """Switch to icons view"""
        if self.view_mode != 'icons':
            self.show_icons_view()
            if getattr(self, '_search_active', False):
                self._display_search_results(getattr(self, '_last_search_results', []))
            else:
                self.load_directory(self.current_path)
        
    def load_directory(self, path):
        """Load directory contents with a non-blocking worker-thread architecture.

        The UI thread:
          1. Validates the path and updates the address bar immediately.
          2. Shows "Loading…" in the status bar.
          3. Spawns a daemon worker to enumerate the filesystem.

        The worker thread:
          • Runs iterdir(), stat(), and hidden-item filtering (all file I/O).
          • Posts the sorted (folders, files) lists back via root.after().
          • Checks _nav_generation before posting; if it changed the user has
            already navigated elsewhere and the result is silently dropped.

        This means the UI never freezes regardless of how fast the user clicks.
        """
        # ── Reset search state immediately ───────────────────────────────
        if getattr(self, '_search_active', False):
            self._search_active = False
            self._last_search_results = []
            try:
                self.search_clear_btn.config(state=tk.DISABLED)
                self.search_status_label.config(
                    text='Add tags → click chip to toggle include / exclude',
                    foreground='gray')
            except Exception:
                pass

        # ── Fast path validation on the UI thread ────────────────────────
        try:
            path = Path(path)
        except Exception:
            return
        if not path.exists():
            messagebox.showerror("Error", f"Path does not exist: {path}")
            return
        if not path.is_dir():
            messagebox.showerror("Error", f"Not a directory: {path}")
            return

        # ── Bump generation counters — cancels any in-flight workers ─────
        self._nav_generation   += 1
        self._thumb_generation += 1          # abort running thumbnail threads
        current_nav = self._nav_generation

        # ── Immediate UI feedback (zero I/O) ─────────────────────────────
        self.current_path = path
        self.path_var.set(str(path))
        self.canvas.yview_moveto(0)
        self.tree.yview_moveto(0)
        self.status_var.set("Loading…")

        # Instrumentation: measure how long the UI-thread portion takes
        ui_start = time.time()

        if hasattr(self, 'controller') and self.controller:
            self.controller.change_directory(str(path))

        show_hidden = self.show_hidden_var.get()

        # ── Worker: all file-system I/O happens here ─────────────────────
        def _enumerate() -> None:
            if self._nav_generation != current_nav:
                return  # user navigated away before we even started

            try:
                # os.scandir() caches stat/type info from the OS directory read,
                # so subsequent is_dir()/is_file() calls cost no extra syscalls.
                raw = [e for e in os.scandir(path) if not e.name.startswith('.') and e.name.lower() != _THUMBS_FOLDER]
            except PermissionError:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Permission denied: {path}"))
                return
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Failed to read directory: {exc}"))
                return

            if self._nav_generation != current_nav:
                return  # stale — user moved on while scandir was running

            # is_dir/is_file on DirEntry are free — stat is cached from the scan
            if not show_hidden:
                raw = [e for e in raw if not self._is_os_hidden_entry(e)]

            folders = sorted(
                [Path(e.path) for e in raw if e.is_dir(follow_symlinks=False)],
                key=lambda x: x.name.lower()
            )
            files = sorted(
                [Path(e.path) for e in raw if e.is_file(follow_symlinks=False)],
                key=lambda x: x.name.lower()
            )

            # DB app-hidden filter runs here in the background, not on the UI thread
            app_hidden: set = set()
            if not show_hidden:
                try:
                    app_hidden = self._get_app_hidden_paths(str(path))
                    if app_hidden:
                        files = [f for f in files if str(f.resolve()) not in app_hidden]
                except Exception:
                    pass

            if self._nav_generation != current_nav:
                return

            self.root.after(0, lambda: _show(folders, files))

        # ── UI callback: render the enumerated results (pure UI work only) ──
        def _show(folders: list, files: list) -> None:
            if self._nav_generation != current_nav:
                return  # user navigated again before callback fired

            if self.view_mode == 'detailed':
                self.load_detailed_view(folders, files)
            else:
                self.load_icons_view(folders, files)

            self.status_var.set(f"{len(folders)} folders, {len(files)} files")

            # Keep side nav tree in sync with the current path
            try:
                self._nav_tree_reveal_path(path)
            except Exception:
                pass

            # Fire one-shot startup hook so the splash knows the dir is loaded
            _hook = getattr(self, '_startup_dir_done_callback', None)
            if _hook is not None:
                self._startup_dir_done_callback = None
                try:
                    _hook()
                except Exception:
                    pass

        threading.Thread(target=_enumerate, daemon=True).start()
        ui_dt = (time.time() - ui_start)
        if ui_dt > 0.05:
            print(f"[UI] load_directory UI path handling took {ui_dt:.3f}s")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # View Options panel
    # ------------------------------------------------------------------

    def _build_navigation_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=4, pady=4)

        self.back_btn = ttk.Button(row, text="← Back", command=self.go_back)
        self.back_btn.pack(side=tk.LEFT, padx=2)
        self.forward_btn = ttk.Button(row, text="→ Forward", command=self.go_forward)
        self.forward_btn.pack(side=tk.LEFT, padx=2)
        self.up_btn = ttk.Button(row, text="↑ Up", command=self.go_up)
        self.up_btn.pack(side=tk.LEFT, padx=2)
        self.home_btn = ttk.Button(row, text="🏠 Home", command=self.go_home)
        self.home_btn.pack(side=tk.LEFT, padx=2)
        self.refresh_btn = ttk.Button(row, text="🔄 Refresh", command=self.refresh)
        self.refresh_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(row, text="Path:").pack(side=tk.LEFT, padx=(2, 2))
        self.path_var = tk.StringVar(value=str(self.current_path))
        self.path_entry = ttk.Entry(row, textvariable=self.path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.path_entry.bind('<Return>', lambda _e: self.navigate_to_path())
        self.path_var.trace_add('write', self._schedule_path_navigation)

    def _build_simple_search_controls(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── filename row ─────────────────────────────────────────────────
        name_row = ttk.Frame(outer)
        name_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(name_row, text='Filename:', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 4))
        self._simple_search_var = tk.StringVar(value=self._ui_state.get('simple_search', ''))
        entry = ttk.Entry(name_row, textvariable=self._simple_search_var, width=28)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        entry.bind('<Return>', lambda _e: self._simple_search_run())
        ttk.Button(name_row, text='Search', command=self._simple_search_run).pack(side=tk.LEFT, padx=2)
        ttk.Button(name_row, text='Clear',  command=self._clear_simple_search).pack(side=tk.LEFT, padx=2)

        # ── tag matrix header ─────────────────────────────────────────────
        hdr = ttk.Frame(outer)
        hdr.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(hdr, text='Filter by tag:', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        ttk.Button(hdr, text='All',  width=4,
                   command=self._tag_matrix_select_all).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(hdr, text='None', width=4,
                   command=self._tag_matrix_deselect_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(hdr, text='↺', width=3,
                   command=self._tag_matrix_refresh).pack(side=tk.LEFT, padx=2)

        # ── scrollable checkbox grid ───────────────────────────────────────
        matrix_outer = ttk.Frame(outer, relief='groove', borderwidth=1)
        matrix_outer.pack(fill=tk.BOTH, expand=True)
        self._tag_matrix_canvas = tk.Canvas(matrix_outer, highlightthickness=0)
        _sb = ttk.Scrollbar(matrix_outer, orient='vertical',
                            command=self._tag_matrix_canvas.yview)
        self._tag_matrix_canvas.configure(yscrollcommand=_sb.set)
        _sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tag_matrix_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._tag_matrix_inner = ttk.Frame(self._tag_matrix_canvas)
        self._tag_matrix_win_id = self._tag_matrix_canvas.create_window(
            (0, 0), window=self._tag_matrix_inner, anchor='nw')
        self._tag_matrix_inner.bind(
            '<Configure>',
            lambda _e: self._tag_matrix_canvas.configure(
                scrollregion=self._tag_matrix_canvas.bbox('all')))
        self._tag_matrix_canvas.bind(
            '<Configure>',
            lambda e: self._tag_matrix_canvas.itemconfig(
                self._tag_matrix_win_id, width=e.width))

        self._tag_matrix_vars: dict = {}
        self._tag_matrix_refresh()

    # ------------------------------------------------------------------
    # Tag-matrix helpers (Search tab)
    # ------------------------------------------------------------------

    def _tag_matrix_refresh(self) -> None:
        """Reload all tags from DB and rebuild the checkbox grid (async)."""
        def _worker() -> None:
            try:
                tags = [r['tag_name'] for r in metadata_store.get_all_tags()]
            except Exception:
                tags = []
            try:
                self.root.after(0, lambda: self._tag_matrix_populate(tags))
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    def _tag_matrix_populate(self, tags: list) -> None:
        """Rebuild checkbox grid from *tags*, preserving existing selections."""
        try:
            inner = self._tag_matrix_inner
        except AttributeError:
            return
        for w in inner.winfo_children():
            w.destroy()
        prev = {t: v.get() for t, v in self._tag_matrix_vars.items()}
        self._tag_matrix_vars = {}
        cols = 3
        for i, tag in enumerate(tags):
            row, col = divmod(i, cols)
            var = tk.BooleanVar(value=prev.get(tag, False))
            cb = ttk.Checkbutton(inner, text=tag, variable=var,
                                 command=self._simple_search_run)
            cb.grid(row=row, column=col, sticky='w', padx=6, pady=1)
            self._tag_matrix_vars[tag] = var
        try:
            inner.update_idletasks()
            self._tag_matrix_canvas.configure(
                scrollregion=self._tag_matrix_canvas.bbox('all'))
        except Exception:
            pass

    def _tag_matrix_select_all(self) -> None:
        for v in self._tag_matrix_vars.values():
            v.set(True)
        self._simple_search_run()

    def _tag_matrix_deselect_all(self) -> None:
        for v in self._tag_matrix_vars.values():
            v.set(False)
        self._simple_search_run()

    def _simple_search_run(self) -> None:
        """Execute combined filename + tag-filter search."""
        filename_q = getattr(self, '_simple_search_var',
                             tk.StringVar()).get().strip().lower()
        included_tags = [t for t, v in getattr(self, '_tag_matrix_vars', {}).items()
                         if v.get()]

        if not filename_q and not included_tags:
            self._clear_simple_search()
            return

        base = self.current_path
        current_gen = self._nav_generation

        def _worker() -> None:
            results: list = []
            try:
                if included_tags:
                    rows = metadata_store.search_images_advanced(
                        included_tags, [], sort_by='name', sort_asc=True)
                    paths = [Path(r['file_path']) for r in rows
                             if Path(r['file_path']).exists()]
                    if filename_q:
                        paths = [p for p in paths if filename_q in p.name.lower()]
                    results = paths
                else:
                    for p in base.rglob('*'):
                        if self._nav_generation != current_gen:
                            return
                        if (p.is_file()
                                and not p.name.startswith('.')
                                and not any(part.lower() == _THUMBS_FOLDER
                                            for part in p.relative_to(base).parts)
                                and p.suffix.lower() in self._IMAGE_EXTS
                                and filename_q in p.name.lower()):
                            results.append(p)
            except Exception:
                results = []

            def _show() -> None:
                if self._nav_generation != current_gen:
                    return
                self._simple_search_results = results
                self._search_mode_var.set('simple')
                tag_info = (f'  [{len(included_tags)} tag(s)]'
                            if included_tags else '')
                self.status_var.set(
                    f'Search{tag_info}: {len(results)} result(s)')
                if self.view_mode == 'detailed':
                    self.load_detailed_view([], results)
                else:
                    self.load_icons_view([], results)

            try:
                self.root.after(0, _show)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _build_view_options_bar(self, parent=None):
        """View tab controls: recompile previews plus gallery/detail toggles."""
        if parent is None:
            parent = self.root
        outer = ttk.Frame(parent, relief='groove', borderwidth=1)
        outer.pack(fill=tk.X, padx=5, pady=(0, 2))

        # ── Header with the only controls we keep in this tab ─────────────
        header = ttk.Frame(outer)
        header.pack(fill=tk.X)

        # View mode toggles (Detailed / Gallery)
        self._view_detailed_btn = ttk.Button(header, text='Detailed', width=10,
                                             command=self.switch_to_detailed)
        self._view_detailed_btn.pack(side=tk.RIGHT, padx=(4, 2))
        self._view_icons_btn = ttk.Button(header, text='Gallery', width=10,
                                          command=self.switch_to_icons)
        self._view_icons_btn.pack(side=tk.RIGHT, padx=(2, 4))

        # Recompile previews button (acts on current path)
        self._recompile_previews_btn = ttk.Button(header, text='Recompile Previews',
                                                  command=self._recompile_previews_from_current_path)
        self._recompile_previews_btn.pack(side=tk.RIGHT, padx=(8, 4))

        ttk.Separator(header, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=6)

        # View Metrics button — surfaces what was only accessible via View menu
        ttk.Button(header, text='Metrics', width=8,
                   command=self.show_metrics).pack(side=tk.RIGHT, padx=(0, 4))

        # Show hidden items toggle — surfaces what was only accessible via View menu
        ttk.Checkbutton(header, text='Show Hidden',
                        variable=self.show_hidden_var,
                        command=self.refresh).pack(side=tk.LEFT, padx=(4, 8))

        # Gallery image height slider (inline with header controls)
        self._zoom_var = tk.IntVar(value=self._view_zoom_override or 140)
        slider_frame = ttk.Frame(header)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 4))

        ttk.Label(slider_frame, text='Gallery size:', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(4, 6))
        zoom_slider = ttk.Scale(slider_frame, from_=72, to=240, orient=tk.HORIZONTAL,
                variable=self._zoom_var, length=280,
                command=self._on_zoom_change)
        zoom_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self._zoom_indicator = ttk.Label(slider_frame, text=f'{int(self._zoom_var.get())} px', font=('Arial', 9), foreground='#555')
        self._zoom_indicator.pack(side=tk.LEFT, padx=(6, 6))

        self._zoom_auto_btn = ttk.Button(slider_frame, text='Auto', width=5,
                 command=self._zoom_auto)
        self._zoom_auto_btn.pack(side=tk.LEFT)

        # Keep the remaining view state simple and fixed.
        self._view_gallery_cols: int = _GALLERY_COLS
        if self._view_zoom_override is None:
            self._view_zoom_override = 0

    def _toggle_view_options(self):
        # Legacy hook kept for compatibility; the View tab no longer collapses.
        return

    def _recompile_previews_from_current_path(self):
        cur = getattr(self, 'current_path', None)
        if cur is None:
            return
        try:
            folders = [p for p in cur.rglob('*') if p.is_dir()
                       and not any(part.lower() == _THUMBS_FOLDER for part in p.relative_to(cur).parts)]
        except Exception:
            folders = []
        # Include the current folder as well
        folders.insert(0, cur)
        if not folders:
            messagebox.showinfo('Recompile Previews', 'No folders found under current path.')
            return
        if not messagebox.askyesno('Recompile Previews',
                                   f'Rebuild previews for {len(folders)} folder(s) under:\n{cur}\n\nThis may be CPU-intensive. Continue?'):
            return

        def _worker():
            for f in folders:
                try:
                    self._rebuild_folder_preview_async(f)
                except Exception:
                    pass
                time.sleep(0.05)
            self.root.after(0, lambda: self.status_var.set(f'Recompiled previews under {cur}'))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_control_tab_changed(self, event=None):
        try:
            if not hasattr(self, '_control_tabs') or not hasattr(self, '_control_tabs_container'):
                return
            sel = self._control_tabs.select()
            if not sel:
                return
            try:
                widget = self._control_tabs.nametowidget(sel)
            except Exception:
                widget = None
            if widget is None:
                return
            # Determine a sensible minimum height per-tab so small tabs remain visible
            try:
                tab_text = self._control_tabs.tab(sel, 'text')
            except Exception:
                tab_text = ''
            min_map = {
                'Search': 84,
                'Advanced Search': 140,
                'View': 84,
                'ML': 120,
            }
            min_h = min_map.get(tab_text, 36)

            # Force layout update so winfo_reqheight() is accurate
            self.root.update_idletasks()
            h = widget.winfo_reqheight()
            # Add a small padding to avoid clipping and enforce the per-tab minimum
            desired = max(min_h, h + 8)
            try:
                self._control_tabs_container.configure(height=desired)
            except Exception:
                pass
            # If the user switched to the View tab, auto-expand the view options
            try:
                if tab_text == 'View' and hasattr(self, '_view_opts_expanded') and self._view_opts_expanded is not None and not self._view_opts_expanded.get():
                    # Show the view options content
                    try:
                        if self._view_opts_content is not None:
                            self._view_opts_content.pack(fill=tk.X)
                    except Exception:
                        pass
                    self._view_opts_expanded.set(True)
                    try:
                        if self._view_opts_toggle is not None:
                            self._view_opts_toggle.config(text='▼ View Options')
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

    def _on_zoom_change(self, _val=None):
        v = int(float(self._zoom_var.get()))
        self._view_zoom_override = v
        self._zoom_indicator.config(text=f'{v} px')
        if self.view_mode == 'icons':
            if hasattr(self, '_gallery_items') and self._gallery_items:
                folders, files = self._gallery_items
                self.load_icons_view(folders, files)

    def _zoom_auto(self):
        self._view_zoom_override = 0
        self._zoom_var.set(0)
        self._zoom_indicator.config(text='auto')
        if self.view_mode == 'icons' and hasattr(self, '_gallery_items') and self._gallery_items:
            folders, files = self._gallery_items
            self.load_icons_view(folders, files)

    def _on_cols_change(self):
        try:
            v = max(2, min(12, int(self._cols_var.get())))
        except (ValueError, tk.TclError):
            return
        self._view_gallery_cols = v
        if self.view_mode == 'icons' and hasattr(self, '_gallery_items') and self._gallery_items:
            folders, files = self._gallery_items
            self.load_icons_view(folders, files)

    def _update_view_toggle_buttons(self):
        """Update the visual state of the view toggle buttons."""
        try:
            if getattr(self, 'view_mode', 'detailed') == 'detailed':
                self._view_detailed_btn.state(['disabled'])
                self._view_icons_btn.state(['!disabled'])
            else:
                self._view_icons_btn.state(['disabled'])
                self._view_detailed_btn.state(['!disabled'])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # MangaDex-style Tag Search
    # ------------------------------------------------------------------
    # Each tag chip cycles: included (+, green) \u2192 excluded (\u2212, red) \u2192 removed.
    # The search executes via search_images_advanced which supports both.
    # ------------------------------------------------------------------

    def _build_search_bar(self, parent=None):
        """Advanced tag-search bar with a single-canvas chip palette."""
        if parent is None:
            parent = self.root
        outer = ttk.Frame(parent, relief='groove', borderwidth=1)
        outer.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 2))
        self._search_bar_frame = outer

        # Row 1: tag-name entry + autocomplete + sort + action buttons
        row1 = ttk.Frame(outer)
        row1.pack(fill=tk.X, padx=4, pady=(3, 1))

        ttk.Label(row1, text='🔍', font=('Arial', 11)).pack(side=tk.LEFT, padx=(0, 4))

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(row1, textvariable=self.search_var, width=28,
                                      font=('Arial', 10))
        self.search_entry.pack(side=tk.LEFT, padx=(0, 4))
        self.search_entry.bind('<Return>', lambda e: self._commit_search_token())
        self.search_entry.bind('<Escape>', lambda e: self._hide_tag_suggestions())
        self.search_var.trace_add('write', self._on_search_entry_change)

        ttk.Button(row1, text='Add tag ↵',
                   command=self._commit_search_token).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(row1, text='Sort:', font=('Arial', 9)).pack(side=tk.LEFT)
        self._sort_var = tk.StringVar(value='name')
        ttk.Combobox(row1, textvariable=self._sort_var, width=8, state='readonly',
                     values=['name', 'date', 'size'],
                     font=('Arial', 9)).pack(side=tk.LEFT, padx=2)

        self._sort_asc_var = tk.BooleanVar(value=True)
        self._sort_dir_btn = ttk.Button(row1, text='↑ Asc', width=6,
                                        command=self._toggle_sort_dir)
        self._sort_dir_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(row1, text='Search',
                   command=self.perform_tag_search).pack(side=tk.LEFT, padx=2)
        self.search_clear_btn = ttk.Button(row1, text='✕ Clear',
                                           command=self.clear_search, state=tk.DISABLED)
        self.search_clear_btn.pack(side=tk.LEFT, padx=2)

        self.search_status_label = ttk.Label(
            row1, text='Click tag = include  ·  again = exclude  ·  again = clear',
            foreground='gray', font=('Arial', 9))
        self.search_status_label.pack(side=tk.LEFT, padx=(10, 0))

        # Row 2: palette filter entry + color legend
        row2 = ttk.Frame(outer)
        row2.pack(fill=tk.X, padx=4, pady=(2, 3))

        ttk.Label(row2, text='Filter:', font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 3))
        self._tag_filter_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self._tag_filter_var, width=24,
                  font=('Arial', 9)).pack(side=tk.LEFT)
        self._tag_filter_var.trace_add('write', lambda *_: self._schedule_palette_redraw())

        ttk.Button(row2, text='Refresh', width=8,
                   command=self._refresh_advanced_tag_sources).pack(side=tk.LEFT, padx=(8, 0))

        legend = ttk.Frame(row2)
        legend.pack(side=tk.RIGHT, padx=(0, 4))
        for _lc, _ll in (('#e8851a', 'Include'), ('#c62828', 'Exclude')):
            tk.Frame(legend, bg=_lc, width=12, height=12).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Label(legend, text=_ll, font=('Arial', 8)).pack(side=tk.LEFT)

        # Canvas palette — all chips drawn as primitives on one canvas
        palette_outer = ttk.Frame(outer)
        palette_outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self._palette_canvas = tk.Canvas(palette_outer, bg='#fafafa',
                                          highlightthickness=0, relief='flat')
        _pal_vsb = ttk.Scrollbar(palette_outer, orient='vertical',
                                  command=self._palette_canvas.yview)
        self._palette_canvas.configure(yscrollcommand=_pal_vsb.set)
        _pal_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._palette_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._palette_canvas.bind('<Configure>',
                                   lambda _e: self._schedule_palette_redraw())
        self._palette_canvas.bind('<Button-1>', self._on_palette_click)
        self._palette_canvas.bind(
            '<MouseWheel>',
            lambda e: self._palette_canvas.yview_scroll(-1 * (e.delta // 120), 'units'))
        self._palette_canvas.bind(
            '<Button-4>', lambda _e: self._palette_canvas.yview_scroll(-1, 'units'))
        self._palette_canvas.bind(
            '<Button-5>', lambda _e: self._palette_canvas.yview_scroll(1, 'units'))

        # Internal state
        self._tag_states:           dict   = {}
        self._palette_all_tags:     list   = []
        self._palette_chip_regions: list   = []
        self._palette_redraw_job:   object = None
        self._palette_font_obj:     object = None
        self._search_active:        bool   = False
        self._last_search_results:  list   = []

        self._suggest_popup:   tk.Toplevel | None = None  # type: ignore[assignment]
        try:
            lbox = tk.Listbox(self.root)
            lbox.pack_forget()
        except Exception:
            lbox = None
        self._suggest_listbox: tk.Listbox | None = lbox  # type: ignore[assignment]

        self._refresh_advanced_tag_sources()

    # Tag palette: data loading

    def _refresh_advanced_tag_sources(self):
        """Reload the full tag list from DB in a background thread."""
        def _worker() -> None:
            try:
                tags = sorted(row['tag_name'] for row in metadata_store.get_all_tags())
            except Exception:
                tags = []
            try:
                self.root.after(0, lambda: self._apply_palette_tags(tags))
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    def _apply_palette_tags(self, tags: list) -> None:
        self._palette_all_tags = tags
        self._redraw_tag_palette()

    # Tag palette: canvas drawing

    def _get_palette_font(self):
        if self._palette_font_obj is None:
            from tkinter import font as _tkfont
            self._palette_font_obj = _tkfont.Font(family='Arial', size=9, weight='bold')
        return self._palette_font_obj

    def _schedule_palette_redraw(self) -> None:
        """Debounce rapid redraws (resize, filter typing)."""
        if self._palette_redraw_job is not None:
            try:
                self.root.after_cancel(self._palette_redraw_job)
            except Exception:
                pass
        self._palette_redraw_job = self.root.after(30, self._redraw_tag_palette)

    def _redraw_tag_palette(self) -> None:
        """Redraw all tag chips on the single palette canvas."""
        self._palette_redraw_job = None
        if not hasattr(self, '_palette_canvas'):
            return
        c = self._palette_canvas
        c.delete('all')
        self._palette_chip_regions = []

        canvas_w = c.winfo_width()
        if canvas_w < 60:
            canvas_w = 400

        filt = getattr(self, '_tag_filter_var', None)
        filt_text = filt.get().lower().strip() if filt else ''
        extra = sorted(t for t in self._tag_states if t not in self._palette_all_tags)
        all_display = sorted(set(self._palette_all_tags) | set(extra))
        if filt_text:
            all_display = [t for t in all_display if filt_text in t.lower()]

        if not all_display:
            msg = 'No tags match.' if filt_text else 'No tags found — click Refresh.'
            c.create_text(10, 12, anchor='nw', text=msg,
                          fill='#999999', font=('Arial', 9))
            c.configure(scrollregion=(0, 0, canvas_w, 36))
            return

        font   = self._get_palette_font()
        CHIP_H = 26
        PAD_X  = 10
        GAP_X  = 5
        GAP_Y  = 5
        RADIUS = 6
        x = y  = 6

        for tag in all_display:
            state = self._tag_states.get(tag, '')
            if state == '+':
                bg, border, fg = '#e8851a', '#c86a00', '#ffffff'
            elif state == '-':
                bg, border, fg = '#c62828', '#9a1c1c', '#ffffff'
            else:
                bg, border, fg = '#ffffff', '#888888', '#222222'

            chip_w = font.measure(tag) + PAD_X * 2

            if x + chip_w > canvas_w - 6 and x > 6:
                x  = 6
                y += CHIP_H + GAP_Y

            x1, y1, x2, y2 = x, y, x + chip_w, y + CHIP_H
            self._draw_palette_chip(c, x1, y1, x2, y2, bg, border, fg, tag, RADIUS, font)
            self._palette_chip_regions.append((x1, y1, x2, y2, tag))
            x += chip_w + GAP_X

        total_h = y + CHIP_H + GAP_Y + 6
        c.configure(scrollregion=(0, 0, canvas_w, total_h))

    @staticmethod
    def _draw_palette_chip(canvas, x1, y1, x2, y2, bg, border, fg, text, r, font):
        """Draw one rounded-rectangle chip on a canvas."""
        pts = [
            x1 + r, y1,      x2 - r, y1,
            x2,     y1,      x2,     y1 + r,
            x2,     y2 - r,  x2,     y2,
            x2 - r, y2,      x1 + r, y2,
            x1,     y2,      x1,     y2 - r,
            x1,     y1 + r,  x1,     y1,
        ]
        canvas.create_polygon(pts, smooth=True, fill=bg, outline=border, width=1.5)
        canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2,
                           text=text, fill=fg, font=font, anchor='center')

    def _on_palette_click(self, event) -> None:
        cx = self._palette_canvas.canvasx(event.x)
        cy = self._palette_canvas.canvasy(event.y)
        for x1, y1, x2, y2, tag in self._palette_chip_regions:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                self._cycle_palette_tag(tag)
                return

    def _cycle_palette_tag(self, tag: str) -> None:
        current = self._tag_states.get(tag, '')
        if current == '':
            self._tag_states[tag] = '+'
        elif current == '+':
            self._tag_states[tag] = '-'
        else:
            self._tag_states.pop(tag, None)
        self._redraw_tag_palette()
        if self._search_active or self._tag_states:
            self.perform_tag_search()


    # \u2500\u2500 sort direction \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _toggle_sort_dir(self):
        asc = not self._sort_asc_var.get()
        self._sort_asc_var.set(asc)
        self._sort_dir_btn.config(text='\u2191 Asc' if asc else '\u2193 Desc')
        if self._search_active:
            self.perform_tag_search()

    # \u2500\u2500 chip management \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _commit_search_token(self):
        """Add the current entry text as a new included-tag chip."""
        raw = self.search_var.get().strip()
        if not raw:
            return
        self._add_tag_chip(raw, '+')
        self.search_var.set('')
        self._hide_tag_suggestions()

    def _add_tag_chip(self, tag: str, state: str = '+'):
        """Add or update a tag. state: '+' = include, '-' = exclude."""
        if not tag:
            return
        self._tag_states[tag] = state
        self._redraw_tag_palette()
        if self._search_active or self._tag_states:
            self.perform_tag_search()

    def _remove_chip(self, tag: str):
        self._tag_states.pop(tag, None)
        self._redraw_tag_palette()
        if self._search_active or self._tag_states:
            self.perform_tag_search()
        elif not self._tag_states:
            self.clear_search()


    # \u2500\u2500 autocomplete popup \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def _on_search_entry_change(self, *_):
        # Debounce to avoid querying the DB on every keystroke.
        try:
            job = getattr(self, '_tag_suggest_job', None)
            if job is not None:
                self.root.after_cancel(job)
        except Exception:
            pass

        def _run():
            try:
                self._show_tag_suggestions(self.search_var.get())
            except Exception:
                pass

        try:
            self._tag_suggest_job = self.root.after(120, _run)
        except Exception:
            # Fallback if after() fails for any reason
            self._show_tag_suggestions(self.search_var.get())

    def _show_tag_suggestions(self, query: str):
        token = query.strip()
        if not token:
            self._hide_tag_suggestions()
            return

        # Cache tag list briefly; on large DBs, fetching all tags repeatedly
        # can cause noticeable UI stutters.
        now = time.time()
        tag_names = None
        try:
            cache = getattr(self, '_tag_names_cache', None)
            cache_ts = float(getattr(self, '_tag_names_cache_ts', 0.0) or 0.0)
            if isinstance(cache, list) and (now - cache_ts) < 2.5:
                tag_names = cache
        except Exception:
            tag_names = None

        if tag_names is None:
            try:
                tag_names = [t['tag_name'] for t in metadata_store.get_all_tags()]
            except Exception:
                tag_names = []
            try:
                self._tag_names_cache = tag_names
                self._tag_names_cache_ts = now
            except Exception:
                pass
        # Exact-start matches first, then substring matches
        starts  = [t for t in tag_names if t.lower().startswith(token.lower())]
        contains = [t for t in tag_names if token.lower() in t.lower() and t not in starts]
        matches  = (starts + contains)[:12]
        if not matches:
            self._hide_tag_suggestions()
            return
        self.search_entry.update_idletasks()
        x = self.search_entry.winfo_rootx()
        y = self.search_entry.winfo_rooty() + self.search_entry.winfo_height()
        h = min(len(matches), 8)
        if self._suggest_popup is None or not self._suggest_popup.winfo_exists():
            popup = tk.Toplevel(self.root)
            popup.wm_overrideredirect(True)
            popup.wm_geometry(f'+{x}+{y}')
            lbox = tk.Listbox(popup, font=('Arial', 10), height=h,
                              selectmode=tk.SINGLE, width=36,
                              relief='solid', borderwidth=1,
                              bg='#ffffff', selectbackground='#0078d4',
                              selectforeground='white')
            lbox.pack(fill=tk.BOTH, expand=True)
            lbox.bind('<Double-1>', self._on_suggestion_select)
            lbox.bind('<Return>',   self._on_suggestion_select)
            self._suggest_popup   = popup
            self._suggest_listbox = lbox
        else:
            self._suggest_popup.wm_geometry(f'+{x}+{y}')
            if self._suggest_listbox is None or not isinstance(self._suggest_listbox, tk.Listbox):
                # Create a listbox attached to the popup if one does not exist
                lbox = tk.Listbox(self._suggest_popup, font=('Arial', 10), height=h,
                                  selectmode=tk.SINGLE, width=36,
                                  relief='solid', borderwidth=1,
                                  bg='#ffffff', selectbackground='#0078d4',
                                  selectforeground='white')
                lbox.pack(fill=tk.BOTH, expand=True)
                lbox.bind('<Double-1>', self._on_suggestion_select)
                lbox.bind('<Return>',   self._on_suggestion_select)
                self._suggest_listbox = lbox
            else:
                self._suggest_listbox.config(height=h)
                self._suggest_listbox.delete(0, tk.END)
            self._suggest_popup.deiconify()
        for m in matches:
            if self._suggest_listbox is not None:
                self._suggest_listbox.insert(tk.END, m)

    def _hide_tag_suggestions(self):
        if self._suggest_popup and self._suggest_popup.winfo_exists():
            self._suggest_popup.withdraw()

    def _on_suggestion_select(self, _event=None):
        if self._suggest_listbox is None:
            return
        sel = self._suggest_listbox.curselection()
        if not sel:
            return
        chosen = self._suggest_listbox.get(sel[0])
        self._hide_tag_suggestions()
        self._add_tag_chip(chosen, '+')
        self.search_var.set('')
        if self._tag_states:
            self.perform_tag_search()

    # \u2500\u2500 search execution \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def perform_tag_search(self):
        """Execute advanced search using current tag chips + sort settings."""
        included = [t for t, s in self._tag_states.items() if s == '+']
        excluded = [t for t, s in self._tag_states.items() if s == '-']

        if not included and not excluded:
            self.clear_search()
            return

        sort_by  = self._sort_var.get()
        sort_asc = self._sort_asc_var.get()
        try:
            results = metadata_store.search_images_advanced(
                included, excluded, sort_by=sort_by, sort_asc=sort_asc)
        except Exception as exc:
            messagebox.showerror('Search Error', f'Tag search failed:\n{exc}')
            return

        self._search_active       = True
        self._last_search_results = results
        self.search_clear_btn.config(state=tk.NORMAL)

        parts = []
        if included:
            parts.append('Included: ' + ', '.join(f'"{t}"' for t in included))
        if excluded:
            parts.append('Excluded: ' + ', '.join(f'"{t}"' for t in excluded))
        self.search_status_label.config(
            text=f'{" | ".join(parts)} \u2014 {len(results)} image(s)',
            foreground='#0078d4')
        self.status_var.set(
            f'Filter active \u2014 {len(results)} image(s) found')

        self._display_search_results(results)
        self._hide_tag_suggestions()

    def _display_search_results(self, results: list):
        """Render search results in the current view mode."""
        files = [Path(r['file_path']) for r in results
                 if Path(r['file_path']).exists() and Path(r['file_path']).is_file()]
        try:
            self.canvas.yview_moveto(0)
        except Exception:
            pass
        try:
            self.tree.yview_moveto(0)
        except Exception:
            pass
        if self.view_mode == 'detailed':
            self.load_detailed_view([], files)
        else:
            self.load_icons_view([], files)

    def clear_search(self):
        """Clear all active filters and reload the current directory."""
        self._search_active       = False
        self._last_search_results = []
        self._tag_states.clear()
        self.search_var.set('')
        self._redraw_tag_palette()
        try:
            self.search_clear_btn.config(state=tk.DISABLED)
            self.search_status_label.config(
                text='Click tag = include  ·  again = exclude  ·  again = clear',
                foreground='gray')
        except Exception:
            pass
        self._hide_tag_suggestions()
        self.load_directory(self.current_path)

    def get_file_type_from_extension(self, path):
        """Determine file type and appropriate icon from extension"""
        if path.is_dir():
            return '📁 Folder', 'folder'
        
        ext = path.suffix.lower()
        
        # Image files
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico', '.jfif']:
            return '🖼️ Image', 'image'
        # Video files
        elif ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']:
            return '🎬 Video', 'video'
        # Audio files
        elif ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma']:
            return '🎵 Audio', 'audio'
        # Document files
        elif ext in ['.pdf']:
            return '📕 PDF', 'document'
        elif ext in ['.doc', '.docx']:
            return '📘 Word', 'document'
        elif ext in ['.xls', '.xlsx']:
            return '📗 Excel', 'document'
        elif ext in ['.ppt', '.pptx']:
            return '📙 PowerPoint', 'document'
        elif ext in ['.txt', '.md', '.log']:
            return '📄 Text', 'document'
        # Archive files
        elif ext in ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2']:
            return '📦 Archive', 'archive'
        # Code files
        elif ext in ['.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go']:
            return '📝 Code', 'code'
        elif ext in ['.html', '.css', '.xml', '.json', '.yaml', '.yml']:
            return '📝 Code', 'code'
        # Executable files
        elif ext in ['.exe', '.dll', '.so', '.app']:
            return '⚙️ Executable', 'executable'
        # Default
        else:
            return '📄 File', 'file'
    
    def create_thumbnail_for_tree(self, path, size=32):
        """Create a small thumbnail for tree view"""
        # Check if already cached
        cache_key = (str(path), size)
        with self._thumb_lock:
            if cache_key in self.thumbnail_cache:
                return self.thumbnail_cache[cache_key]
        
        try:
            t0 = time.time()
            ext = path.suffix.lower()
            
            # For image files, create actual thumbnail
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico', '.jfif']:
                raw = Image.open(path)
                try:
                    img = ImageOps.exif_transpose(raw)
                except Exception:
                    img = raw

                # Convert RGBA to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background

                # Resize to thumbnail size
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                raw.close()

                # Cache the thumbnail
                with self._thumb_lock:
                    self._put_thumbnail_cache(cache_key, photo)
                dt = time.time() - t0
                if dt > 0.05:
                    print(f"[UI] create_thumbnail_for_tree for {path.name} took {dt:.3f}s")
                return photo
            else:
                # For non-image files, create a colored square with file type indicator
                img = Image.new('RGB', (size, size), (240, 240, 240))
                photo = ImageTk.PhotoImage(img)
                img.close()
                with self._thumb_lock:
                    self._put_thumbnail_cache(cache_key, photo)
                dt = time.time() - t0
                if dt > 0.05:
                    print(f"[UI] create_thumbnail_for_tree (non-image) took {dt:.3f}s")
                return photo

        except Exception as e:
            # If thumbnail creation fails, return a default icon
            img = Image.new('RGB', (size, size), (200, 200, 200))
            photo = ImageTk.PhotoImage(img)
            img.close()
            return photo
    
    def load_detailed_view(self, folders, files):
        """Load items in detailed view.

        Items are inserted immediately without thumbnails so the UI stays
        responsive.  A background thread then loads thumbnails one-by-one
        and updates the tree via root.after(), stopping automatically if the
        user navigates to a different folder before it finishes.
        """
        # Bump generation so any running thumbnail thread for the old dir aborts.
        self._thumb_generation += 1
        current_gen = self._thumb_generation

        detailed_ui_start = time.time()

        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Keep thumbnail-reference dict small
        if len(self.tree_thumbnails) > 500:
            self.tree_thumbnails.clear()

        # --- Phase 1: insert all items synchronously WITHOUT thumbnails ---
        # This is O(n) in stat() calls only – fast even for large folders.
        image_items: list[tuple] = []  # (item_id, path) pairs that need a thumbnail

        all_entries = [(f, *self.get_file_type_from_extension(f)) for f in folders] + \
                      [(f, *self.get_file_type_from_extension(f)) for f in files]

        for path, file_type, category in all_entries:
            item_id = self.add_tree_item(path, file_type, category, skip_thumbnail=True)
            if category == 'image' and path.is_file() and item_id:
                image_items.append((item_id, path))

        if not image_items:
            return

        # --- Phase 2: load thumbnails on a background thread ---
        def _load_thumbs():
            for item_id, path in image_items:
                if self._thumb_generation != current_gen:
                    return  # directory changed – abort

                cache_key = (str(path), 24)
                with self._thumb_lock:
                    cached = self.thumbnail_cache.get(cache_key)
                if cached is not None:
                    # Already have a PhotoImage – just schedule the tree update.
                    def _apply_cached(iid=item_id, ph=cached, p=path):
                        if self._thumb_generation != current_gen:
                            return
                        try:
                            self.tree.item(iid, image=ph)
                            self.tree_thumbnails[str(p)] = ph
                        except Exception:
                            pass
                    self.root.after(0, _apply_cached)
                    continue

                # Build PIL image entirely off the main thread
                try:
                    img = Image.open(path)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = bg
                    img.thumbnail((24, 24), Image.Resampling.LANCZOS)
                except Exception:
                    continue

                # Create PhotoImage and update tree on the main thread.
                def _apply(iid=item_id, pil=img, ck=cache_key, p=path):
                    if self._thumb_generation != current_gen:
                        return
                    try:
                        ph = ImageTk.PhotoImage(pil)
                        with self._thumb_lock:
                            self._put_thumbnail_cache(ck, ph)
                        self.tree.item(iid, image=ph)
                        self.tree_thumbnails[str(p)] = ph
                    except Exception:
                        pass
                self.root.after(0, _apply)

        threading.Thread(target=_load_thumbs, daemon=True).start()
        detailed_ui_dt = (time.time() - detailed_ui_start)
        if detailed_ui_dt > 0.05:
            print(f"[UI] load_detailed_view UI work took {detailed_ui_dt:.3f}s")
    
    _IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.jfif'})

    def reset_folder_previews(self):
        """Delete all cached .folder_preview.png files under the current root folder,
        clear in-memory collage cache entries, and refresh the view."""
        root_path = self.current_path
        deleted = 0
        errors = 0
        for preview_file in root_path.rglob(_PREVIEW_FILENAME):
            try:
                preview_file.unlink()
                deleted += 1
            except Exception:
                errors += 1

        # Also purge the in-memory thumbnail cache of any collage entries
        with self._thumb_lock:
            stale_keys = [k for k in list(self.thumbnail_cache.keys())
                          if isinstance(k, tuple) and len(k) == 3 and k[0] == 'folder_collage']
            for k in stale_keys:
                try:
                    del self.thumbnail_cache[k]
                except KeyError:
                    pass

        msg = f"Deleted {deleted} folder preview file(s)."
        if errors:
            msg += f" ({errors} could not be deleted.)"
        messagebox.showinfo("Reset Folder Previews", msg)
        # Clear saved preview snapshots so previews will be rebuilt
        try:
            self._app_state['folder_previews'] = {}
            self._mark_app_state_dirty()
        except Exception:
            pass
        self.load_directory(self.current_path)

    def _first_image_in(self, folder: Path, app_hidden: set) -> 'Path | None':
        """BFS within *folder* to find the first non-hidden image file."""
        queue = [folder]
        while queue:
            current = queue.pop(0)
            try:
                items = sorted(current.iterdir(), key=lambda x: x.name.lower())
            except Exception:
                continue
            subdirs = []
            for item in items:
                if item.name.startswith('.'):
                    continue
                if self._is_os_hidden(item):
                    continue
                if item.name == _PREVIEW_FILENAME:
                    continue
                if item.is_dir() and item.name.lower() == _THUMBS_FOLDER:
                    continue
                if item.is_file() and item.suffix.lower() in self._IMAGE_EXTS:
                    if str(item.resolve()) not in app_hidden:
                        return item
                elif item.is_dir():
                    subdirs.append(item)
            queue = subdirs + queue
        return None

    def _collect_folder_preview_images(self, folder_path: Path, count: int = 4) -> list:
        """Return *count* image paths for the folder preview collage.

        Strategy:
        - If the folder has no subfolders, use direct images from this folder.
        - If it has subfolders, take 1 image per subfolder (up to *count*).
          Any remaining slots are filled with direct images from this folder
          (not cycling — direct images from the current folder are preferred
          over cycling through subfolders again).
        """
        app_hidden = self._get_app_hidden_paths(str(folder_path))

        # Collect direct non-hidden images from this folder
        def _direct_images():
            imgs: list = []
            try:
                direct = sorted(folder_path.iterdir(), key=lambda x: x.name.lower())
            except Exception:
                return imgs
            for item in direct:
                if item.name.startswith('.'):
                    continue
                if self._is_os_hidden(item):
                    continue
                if item.name == _PREVIEW_FILENAME:
                    continue
                if item.is_file() and item.suffix.lower() in self._IMAGE_EXTS:
                    if str(item.resolve()) not in app_hidden:
                        imgs.append(item)
            return imgs

        # Collect direct non-hidden subfolders
        try:
            subfolders = sorted(
                [f for f in folder_path.iterdir()
                 if f.is_dir() and not f.name.startswith('.') and not self._is_os_hidden(f) and f.name != _PREVIEW_FILENAME and f.name.lower() != _THUMBS_FOLDER],
                key=lambda x: x.name.lower()
            )
        except Exception:
            subfolders = []

        # Take at most the first *count* subfolders as source buckets
        buckets = subfolders[:count]

        if not buckets:
            # No subfolders — use images directly in this folder
            return _direct_images()[:count]

        # One image per subfolder (no cycling)
        results: list = []
        for b in buckets:
            if len(results) >= count:
                break
            img = self._first_image_in(b, app_hidden)
            if img is not None:
                results.append(img)

        # Fill remaining slots with direct images from this folder
        if len(results) < count:
            for img in _direct_images():
                if len(results) >= count:
                    break
                if img not in results:
                    results.append(img)

        return results

    @staticmethod
    def _zoom_fill(img: 'Image.Image', size: int, zoom: float = 1.0) -> 'Image.Image':
        """Zoom-crop *img* to fill a *size*×*size* square.

        *zoom* > 1.0 scales the image further before cropping, producing a
        more zoomed-in (tighter crop) result.
        """
        w, h = img.size
        # Scale so the shorter side equals size * zoom, then crop to size
        scale = (size * zoom) / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Centre-crop to the requested size
        left = (new_w - size) // 2
        top  = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def _compute_folder_snapshot(self, folder_path: Path, limit: int = 20) -> str:
        """Compute a lightweight snapshot/hash summarizing image set in *folder_path*.

        We take up to *limit* images (sorted by name) and concat their path+mtime
        then return a hex SHA256 digest. This is used to validate cached previews.
        """
        try:
            parts = []
            items = sorted(folder_path.iterdir(), key=lambda x: x.name.lower())
            cnt = 0
            for it in items:
                if cnt >= limit:
                    break
                if it.name.startswith('.'):
                    continue
                if it.name == _PREVIEW_FILENAME:
                    continue
                if it.is_file() and it.suffix.lower() in self._IMAGE_EXTS:
                    try:
                        m = int(it.stat().st_mtime)
                    except Exception:
                        m = 0
                    parts.append(f"{str(it.resolve())}:{m}")
                    cnt += 1
            if not parts:
                return ''
            h = hashlib.sha256()
            h.update('\n'.join(parts).encode('utf-8'))
            return h.hexdigest()
        except Exception:
            return ''

    def _invalidate_folder_preview_state(self, folder_path: Path) -> None:
        """Remove saved preview state so the next read rebuilds the collage."""
        try:
            previews = self._app_state.get('folder_previews', {})
            previews.pop(str(folder_path), None)
            self.root.after(0, self._mark_app_state_dirty)
        except Exception:
            pass

    def _should_auto_rebuild_folder_preview(self, folder_path: Path, preview_count: int | None = None) -> bool:
        """Return True when a folder preview should be rebuilt automatically.

        Policy:
        - Small folders (<4 images) may auto-rebuild on open if the preview is missing
          or stale, because the work is cheap.
        - Larger folders should keep whatever preview is already on disk and only
          rebuild when the user explicitly requests it.
        """
        try:
            if preview_count is None:
                preview_count = len(self._collect_folder_preview_images(folder_path, 4))
            return preview_count < 4
        except Exception:
            return False

    def _rebuild_folder_preview_async(self, folder_path: Path) -> None:
        """Rebuild a folder preview on a background thread and update caches."""
        folder_path = Path(folder_path)

        def _worker() -> None:
            try:
                preview_file = folder_path / _PREVIEW_FILENAME
                if not folder_path.exists() or not folder_path.is_dir():
                    return

                # Build at storage resolution so the next startup can reuse it.
                collage = self._build_folder_collage(folder_path, _PREVIEW_RESOLUTION)
                try:
                    preview_file.parent.mkdir(parents=True, exist_ok=True)
                    collage.save(str(preview_file), 'PNG', optimize=True)
                except Exception:
                    return

                try:
                    if platform.system() == 'Windows':
                        import ctypes
                        ctypes.windll.kernel32.SetFileAttributesW(str(preview_file), 0x02)
                except Exception:
                    pass

                # Clear cached folder collages for this folder so UI redraws cleanly.
                with self._thumb_lock:
                    stale_keys = [k for k in list(self.thumbnail_cache.keys())
                                  if isinstance(k, tuple)
                                  and len(k) == 3
                                  and k[0] == 'folder_collage'
                                  and k[1] == str(folder_path)]
                    for key in stale_keys:
                        try:
                            del self.thumbnail_cache[key]
                        except KeyError:
                            pass

                if self.current_path == folder_path:
                    self.root.after(0, self.refresh)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _build_folder_collage(self, folder_path: Path, thumb_size: int) -> 'Image.Image':
        """Return a thumb_size×thumb_size preview image for *folder_path*.

        Strategy:
        1. If a valid `.folder_preview.png` exists inside the folder (not stale),
           load and scale it to *thumb_size*.
        2. Otherwise build a fresh 2×2 collage at ``_PREVIEW_RESOLUTION``, save
           it as `.folder_preview.png` (hidden on Windows), then return scaled.
        """
        from PIL import ImageDraw
        import platform as _platform

        preview_file = folder_path / _PREVIEW_FILENAME
        preview_images = self._collect_folder_preview_images(folder_path, 4)
        auto_rebuild = self._should_auto_rebuild_folder_preview(folder_path, len(preview_images))

        # --- Try to load an existing, validated static preview ---
        try:
            snapshot = self._compute_folder_snapshot(folder_path)
            fp = self._app_state.get('folder_previews', {}).get(str(folder_path), {})
            stored_snapshot = fp.get('snapshot', '')
            if preview_file.exists() and snapshot and stored_snapshot == snapshot:
                try:
                    img = Image.open(preview_file).convert('RGB')
                    img = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    return img
                except Exception:
                    pass
            elif preview_file.exists() and not auto_rebuild:
                # Larger folders keep the existing on-disk preview until the
                # user manually triggers a rebuild.
                try:
                    img = Image.open(preview_file).convert('RGB')
                    img = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    return img
                except Exception:
                    pass
        except Exception:
            pass

        # --- Build fresh collage at the fixed storage resolution ---
        res  = _PREVIEW_RESOLUTION
        # If there is no existing on-disk preview and the folder has >=4 images,
        # avoid expensive full rebuild at startup — return a lightweight
        # placeholder so the UI remains responsive. The user can trigger a
        # full recompile via the View tab button.
        try:
            if (not preview_file.exists()) and not auto_rebuild:
                # Quick placeholder: neutral background with folder count badge
                ph = Image.new('RGB', (res, res), (220, 220, 220))
                try:
                    d = ImageDraw.Draw(ph)
                    txt = f"{len(preview_images)} images"
                    d.text((10, 10), txt, fill=(80, 80, 80))
                except Exception:
                    pass
                return ph.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        except Exception:
            pass
        collage = Image.new('RGB', (res, res), (210, 210, 210))

        def _paste(img_path: Path, box: tuple[int, int, int, int]) -> None:
            try:
                img = Image.open(img_path).convert('RGB')
                tile_w = box[2] - box[0]
                tile_h = box[3] - box[1]
                tile_size = max(tile_w, tile_h)
                tile = self._zoom_fill(img, tile_size, zoom=_PREVIEW_ZOOM)
                if tile.size != (tile_w, tile_h):
                    tile = tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                collage.paste(tile, box[:2])
            except Exception:
                pass

        if len(preview_images) >= 4:
            half = res // 2
            boxes = [(0, 0, half, half), (half, 0, res, half), (0, half, half, res), (half, half, res, res)]
        elif len(preview_images) == 3:
            top_h = res // 2
            bottom_h = res - top_h
            bottom_w = res // 2
            boxes = [(0, 0, res, top_h), (0, top_h, bottom_w, res), (bottom_w, top_h, res, res)]
        elif len(preview_images) == 2:
            half = res // 2
            boxes = [(0, 0, res, half), (0, half, res, res)]
        elif len(preview_images) == 1:
            boxes = [(0, 0, res, res)]
        else:
            boxes = []

        for img_path, box in zip(preview_images, boxes):
            _paste(img_path, box)

        # No divider lines — tiles butt flush against each other
        draw = ImageDraw.Draw(collage)

        # Folder badge — bottom-left corner
        bs = max(res // 5, 16)
        bx, by = 3, res - bs - 3
        draw.rectangle([bx, by + bs // 5, bx + bs, by + bs],
                       fill=(255, 200, 50), outline=(140, 100, 0), width=1)
        draw.rectangle([bx, by, bx + bs // 2, by + bs // 5 + 2],
                       fill=(255, 200, 50), outline=(140, 100, 0), width=1)

        # --- Save static preview inside the folder ---
        try:
            collage.save(str(preview_file), 'PNG', optimize=True)
            if _platform.system() == 'Windows':
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetFileAttributesW(str(preview_file), 0x02)
                except Exception:
                    pass
        except Exception:
            pass  # read-only folder — preview still works in-memory

        # Update state with the newly created snapshot so future startups can skip rebuilds
        try:
            snapshot = self._compute_folder_snapshot(folder_path)
            fp = self._app_state.setdefault('folder_previews', {})
            fp[str(folder_path)] = {'snapshot': snapshot, 'preview_mtime': int(preview_file.stat().st_mtime) if preview_file.exists() else 0}
            # schedule debounced save — avoids one write per collage in a folder with many subfolders
            self.root.after(0, self._mark_app_state_dirty)
        except Exception:
            pass

        return collage.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)

    def _gallery_thumb_size(self) -> int:
        """Return thumbnail size respecting the View Options zoom override."""
        override = getattr(self, '_view_zoom_override', 0)
        if override and override >= 64:
            return override
        w = self.canvas.winfo_width()
        if w < 20:
            w = 600
        cols = getattr(self, '_view_gallery_cols', _GALLERY_COLS)
        gap_total = _GALLERY_GAP * (cols + 1)
        return max(40, (w - gap_total) // cols)

    def _relayout_gallery(self):
        """Re-render the gallery when canvas width changes enough to matter."""
        self._gallery_relayout_id = None
        if not hasattr(self, '_gallery_items') or self._gallery_items is None:
            return
        self._layout_gallery_flow()

    def _schedule_gallery_flow_layout(self):
        """Debounce flow layout recalculation for the masonry-like gallery."""
        try:
            if hasattr(self, '_gallery_relayout_id') and self._gallery_relayout_id:
                self.root.after_cancel(self._gallery_relayout_id)
        except Exception:
            pass
        self._gallery_relayout_id = self.root.after(30, self._layout_gallery_flow)

    def _layout_gallery_flow(self):
        """Place gallery items in a wrapping flow layout using their actual widths."""
        self._gallery_relayout_id = None
        if not hasattr(self, '_gallery_flow_items') or not self._gallery_flow_items:
            try:
                self.canvas.configure(scrollregion=self.canvas.bbox('all'))
            except Exception:
                pass
            return

        try:
            self.root.update_idletasks()
            canvas_width = max(1, int(self.canvas.winfo_width()))
        except Exception:
            canvas_width = 800

        gap = _GALLERY_GAP
        left_pad = gap // 2
        right_limit = max(1, canvas_width - left_pad)
        x = left_pad
        y = left_pad
        row_height = 0

        for frame in self._gallery_flow_items:
            try:
                frame.update_idletasks()
                item_w = max(1, int(frame.winfo_reqwidth()))
                item_h = max(1, int(frame.winfo_reqheight()))
            except Exception:
                item_w = getattr(self, '_gallery_last_thumb_size', 120)
                item_h = item_w + 24

            if x > left_pad and (x + item_w) > right_limit:
                x = left_pad
                y += row_height + gap
                row_height = 0

            try:
                frame.place(x=x, y=y)
            except Exception:
                pass

            x += item_w + gap
            row_height = max(row_height, item_h)

        total_h = y + row_height + gap
        try:
            self.icons_container.configure(width=canvas_width, height=total_h)
        except Exception:
            pass
        try:
            self.canvas.configure(scrollregion=(0, 0, canvas_width, total_h))
        except Exception:
            pass

    def load_icons_view(self, folders, files):
        # Store items so the canvas-resize handler can re-layout.
        self._gallery_items = (folders, files)

        # Bump generation so any running thumbnail thread aborts.
        self._thumb_generation += 1
        current_gen = self._thumb_generation
        current_nav = self._nav_generation

        # Compute thumbnail size to fill the canvas width with _GALLERY_COLS columns.
        thumb_size = self._gallery_thumb_size()
        self._gallery_last_thumb_size = thumb_size

        icons_ui_start = time.time()

        # Clear existing icon buttons
        for widget in self.icons_container.winfo_children():
            widget.destroy()
        self.icon_buttons = []
        self.image_references = []  # Clear image references
        self._gallery_flow_items = []

        all_items = folders + files
        pending_thumbs:  list[tuple] = []
        pending_folders: list[tuple] = []
        deferred_thumbs: list[tuple] = []

        _BATCH = 4
        LOOKAHEAD = max(120, _BATCH * 5)
        load_queue = queue.Queue()
        workers_done = {'thumbs': False, 'folders': False}

        def _process_image(lbl, path):
            """Load and resize a single image. Returns a PIL image or None.
            Tries OpenCV first (3-5× faster), falls back to Pillow."""
            if self._thumb_generation != current_gen:
                return None

            def _fit_to_height(img: 'Image.Image', target_h: int) -> 'Image.Image':
                """Resize an image to a fixed height while preserving aspect ratio."""
                try:
                    w, h = img.size
                    if h <= 0:
                        return img
                    if h == target_h:
                        return img
                    scale = target_h / float(h)
                    new_w = max(1, int(round(w * scale)))
                    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)
                except Exception:
                    return img
            try:
                # Check LMDB first (if available), then disk cache. Include file mtime in keys.
                key = hashlib.sha256(str(Path(path).resolve()).encode('utf-8')).hexdigest()
                try:
                    mtime = int(Path(path).stat().st_mtime)
                except Exception:
                    mtime = 0
                cache_key_str = f"{key}_{thumb_size}_{mtime}_v{_THUMB_CACHE_VERSION}"
                cache_key = cache_key_str.encode('utf-8')
                try:
                    if self._lmdb_env is not None:
                        with self._lmdb_env.begin(write=False) as txn:
                            data = txn.get(cache_key)
                            if data is not None:
                                img = Image.open(_io.BytesIO(data))
                                img.load()
                                if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico', '.jfif'}:
                                    img = _fit_to_height(img, thumb_size)
                                return img
                except Exception:
                    pass
                try:
                    _ext = _THUMB_CACHE_FORMAT.lower()
                    cache_path = self.disk_thumbnail_dir / f"{cache_key_str}.{_ext}"
                    if cache_path.exists():
                        img = Image.open(cache_path)
                        img.load()
                        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico', '.jfif'}:
                            img = _fit_to_height(img, thumb_size)
                        return img
                except Exception:
                    pass
                # Try OpenCV first (fast path)
                try:
                    import cv2
                    img_cv = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if img_cv is not None:
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        # Respect EXIF orientation (cv2 ignores it by default).
                        try:
                            _pil_tmp = ImageOps.exif_transpose(Image.fromarray(img_cv))
                            img_cv = None          # release original
                            pil_img = _pil_tmp
                        except Exception:
                            pil_img = Image.fromarray(img_cv)
                        pil_img = _fit_to_height(pil_img, thumb_size)
                        return pil_img
                except (ImportError, Exception):
                    pass  # Fall through to Pillow
                
                # Pillow fallback
                img = Image.open(path)
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    bg.paste(
                        img,
                        mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None
                    )
                    img = bg
                img = _fit_to_height(img, thumb_size)
                # Attempt to write to LMDB (preferred) and disk cache (fallback)
                try:
                    buf = _io.BytesIO()
                    _fmt = _THUMB_CACHE_FORMAT
                    _save_kwargs: dict = {'format': _fmt, 'quality': 88}
                    if _fmt == 'WEBP':
                        _save_kwargs['method'] = 4   # compression speed/ratio balance
                    elif _fmt == 'JPEG':
                        _save_kwargs['subsampling'] = 0
                    img.convert('RGB').save(buf, **_save_kwargs)
                    data = buf.getvalue()
                    _ext = _fmt.lower()
                    # Serialize LMDB and disk writes to avoid concurrent writers
                    with self._thumb_lock:
                        if self._lmdb_env is not None:
                            try:
                                with self._lmdb_env.begin(write=True) as txn:
                                    txn.put(cache_key, data)
                            except Exception:
                                pass
                        try:
                            cache_path = self.disk_thumbnail_dir / f"{cache_key_str}.{_ext}"
                            tmp = cache_path.with_suffix('.tmp')
                            with open(tmp, 'wb') as f:
                                f.write(data)
                            try:
                                tmp.replace(cache_path)
                            except Exception:
                                tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                except Exception:
                    pass
                return img
            except Exception:
                return None

        def _process_folder_collage(path):
            """Build a folder collage and return a PIL image or None."""
            if self._thumb_generation != current_gen:
                return None
            try:
                return self._build_folder_collage(path, thumb_size)
            except Exception:
                return None

        def _drain_load_queue() -> None:
            if self._thumb_generation != current_gen:
                return
            processed = 0
            dq_t0 = time.time()
            layout_dirty = False
            # Process fewer PhotoImage creations per tick to avoid jank
            # lowered from 8 -> 4 and slightly increased delay between ticks
            while processed < 4:
                try:
                    kind, lbl, cache_key, payload = load_queue.get_nowait()
                except queue.Empty:
                    break

                if kind == 'cached':
                    ph = payload
                    try:
                        lbl.config(image=ph, text='')
                        lbl.image = ph  # type: ignore
                        self.image_references.append(ph)
                        layout_dirty = True
                    except Exception:
                        pass
                elif payload is not None:
                    try:
                        ph = ImageTk.PhotoImage(payload)
                        with self._thumb_lock:
                            self._put_thumbnail_cache(cache_key, ph)
                        lbl.config(image=ph, text='')
                        lbl.image = ph  # type: ignore
                        self.image_references.append(ph)
                        layout_dirty = True
                    except Exception:
                        pass
                processed += 1

            # Log if draining is taking long or queue remains large
            dq_dt = time.time() - dq_t0
            try:
                qsize = load_queue.qsize()
            except Exception:
                qsize = -1
            if dq_dt > 0.02 or qsize > 50:
                print(f"[DRain] processed={processed} dt={dq_dt:.3f}s qsize={qsize} thumbs_done={workers_done['thumbs']} folders_done={workers_done['folders']}")

            if layout_dirty:
                self._schedule_gallery_flow_layout()

            if not all(workers_done.values()) or not load_queue.empty():
                self.root.after(25, _drain_load_queue)

        def _load_thumbs() -> None:
            with self._thumb_lock:
                running_gen = getattr(self, '_thumbs_running_gen', None)
                if running_gen == current_gen:
                    print("[_load_thumbs] workers already running for this generation; skipping")
                    return
                self._thumbs_running_gen = current_gen

            def _thumb_worker() -> None:
                t0 = time.time()
                try:
                    total = len(pending_thumbs)
                except Exception:
                    total = 0
                print(f"[_thumb_worker] starting {total} tasks (gen={current_gen})")
                
                # Sliding window: keep at most 8 tasks in flight to prevent queue explosion
                from concurrent.futures import wait, FIRST_COMPLETED
                executor = getattr(self, '_thumb_executor', None)
                if executor is None:
                    # Fallback: local executor
                    executor = ThreadPoolExecutor(max_workers=4)
                    local_executor = True
                else:
                    local_executor = False
                in_flight = {}  # future -> (lbl, cache_key, path)
                pending_idx = 0

                # Submit initial batch (limited to 8)
                while pending_idx < len(pending_thumbs) and len(in_flight) < 8:
                    if self._thumb_generation != current_gen:
                        print("[_thumb_worker] generation changed; aborting")
                        if local_executor:
                            try:
                                executor.shutdown(wait=False)
                            except Exception:
                                pass
                        return
                    lbl, path = pending_thumbs[pending_idx]
                    cache_key = (str(path), thumb_size)
                    with self._thumb_lock:
                        cached = self.thumbnail_cache.get(cache_key)
                    if cached is not None:
                        load_queue.put(('cached', lbl, cache_key, cached))
                    else:
                        fut = executor.submit(_process_image, lbl, path)
                        in_flight[fut] = (lbl, cache_key, path)
                    pending_idx += 1

                # Drain futures; allow adding new ones while running.
                try:
                    while in_flight:
                        if self._thumb_generation != current_gen:
                            print("[_thumb_worker] generation changed during results; aborting")
                            return

                        done, _pending = wait(
                            set(in_flight.keys()),
                            return_when=FIRST_COMPLETED,
                            timeout=0.5,
                        )
                        for future in done:
                            lbl, cache_key, path = in_flight.pop(future)
                            try:
                                res = future.result()
                            except Exception as e:
                                res = None
                                try:
                                    print(f"[_thumb_worker] exception on {path.name}: {e}")
                                except Exception:
                                    pass
                            load_queue.put(('thumb', lbl, cache_key, res))

                        # Submit next tasks if any remain
                        while pending_idx < len(pending_thumbs) and len(in_flight) < 8:
                            if self._thumb_generation != current_gen:
                                return
                            lbl2, path2 = pending_thumbs[pending_idx]
                            cache_key2 = (str(path2), thumb_size)
                            with self._thumb_lock:
                                cached2 = self.thumbnail_cache.get(cache_key2)
                            if cached2 is not None:
                                load_queue.put(('cached', lbl2, cache_key2, cached2))
                            else:
                                fut2 = executor.submit(_process_image, lbl2, path2)
                                in_flight[fut2] = (lbl2, cache_key2, path2)
                            pending_idx += 1
                finally:
                    if local_executor:
                        try:
                            executor.shutdown(wait=False)
                        except Exception:
                            pass

                workers_done['thumbs'] = True
                print(f"[_thumb_worker] finished {total} tasks in {time.time()-t0:.3f}s")
                # clear running flag when thumbs finished (folders may still run)
                with self._thumb_lock:
                    if getattr(self, '_thumbs_running_gen', None) == current_gen:
                        self._thumbs_running_gen = None

            def _folder_worker() -> None:
                t0 = time.time()
                try:
                    total = len(pending_folders)
                except Exception:
                    total = 0
                print(f"[_folder_worker] starting {total} tasks (gen={current_gen})")
                
                # Sliding window: keep at most 4 tasks in flight
                from concurrent.futures import wait, FIRST_COMPLETED
                executor = getattr(self, '_folder_executor', None)
                if executor is None:
                    executor = ThreadPoolExecutor(max_workers=2)
                    local_executor = True
                else:
                    local_executor = False
                in_flight = {}  # future -> (lbl, cache_key, path)
                pending_idx = 0

                # Submit initial batch (limited to 4)
                while pending_idx < len(pending_folders) and len(in_flight) < 4:
                    if self._thumb_generation != current_gen:
                        print("[_folder_worker] generation changed; aborting")
                        if local_executor:
                            try:
                                executor.shutdown(wait=False)
                            except Exception:
                                pass
                        return
                    lbl, path = pending_folders[pending_idx]
                    cache_key = ('folder_collage', str(path), thumb_size)
                    with self._thumb_lock:
                        cached = self.thumbnail_cache.get(cache_key)
                    if cached is not None:
                        load_queue.put(('cached', lbl, cache_key, cached))
                    else:
                        fut = executor.submit(_process_folder_collage, path)
                        in_flight[fut] = (lbl, cache_key, path)
                    pending_idx += 1

                try:
                    while in_flight:
                        if self._thumb_generation != current_gen:
                            print("[_folder_worker] generation changed during results; aborting")
                            return

                        done, _pending = wait(
                            set(in_flight.keys()),
                            return_when=FIRST_COMPLETED,
                            timeout=0.5,
                        )
                        for future in done:
                            lbl, cache_key, path = in_flight.pop(future)
                            try:
                                res = future.result()
                            except Exception as e:
                                res = None
                                try:
                                    print(f"[_folder_worker] exception on {path.name}: {e}")
                                except Exception:
                                    pass
                            load_queue.put(('folder', lbl, cache_key, res))

                        while pending_idx < len(pending_folders) and len(in_flight) < 4:
                            if self._thumb_generation != current_gen:
                                return
                            lbl2, path2 = pending_folders[pending_idx]
                            cache_key2 = ('folder_collage', str(path2), thumb_size)
                            with self._thumb_lock:
                                cached2 = self.thumbnail_cache.get(cache_key2)
                            if cached2 is not None:
                                load_queue.put(('cached', lbl2, cache_key2, cached2))
                            else:
                                fut2 = executor.submit(_process_folder_collage, path2)
                                in_flight[fut2] = (lbl2, cache_key2, path2)
                            pending_idx += 1
                finally:
                    if local_executor:
                        try:
                            executor.shutdown(wait=False)
                        except Exception:
                            pass

                workers_done['folders'] = True
                print(f"[_folder_worker] finished {total} tasks in {time.time()-t0:.3f}s")
                with self._thumb_lock:
                    if getattr(self, '_thumbs_running_gen', None) == current_gen:
                        self._thumbs_running_gen = None

            threading.Thread(target=_thumb_worker, daemon=True).start()
            threading.Thread(target=_folder_worker, daemon=True).start()
            self.root.after(5, _drain_load_queue)

        # Define _start_thumbs after _load_thumbs so it can reference it
        def _prioritize_visible_items():
            """Reorder pending lists so visible items are processed first."""
            try:
                # Ensure layout info is up-to-date
                self.root.update_idletasks()
                canvas_y0 = int(self.canvas.canvasy(0))
                canvas_h = int(self.canvas.winfo_height())
                canvas_y1 = canvas_y0 + canvas_h

                def visible_key(entry):
                    lbl, path = entry
                    try:
                        y = int(lbl.winfo_y())
                        h = int(lbl.winfo_height())
                        # within or near viewport -> lower sort key
                        if (y + h) >= canvas_y0 and y <= canvas_y1:
                            return 0
                        # small distance penalty for near-visible items
                        if y > canvas_y1:
                            return y - canvas_y1
                        return canvas_y0 - (y + h)
                    except Exception:
                        return 999999

                pending_thumbs.sort(key=visible_key)
                pending_folders.sort(key=visible_key)
            except Exception:
                pass

        def _warm_visible_cache():
            """Warm LMDB/disk cache for the first visible + lookahead items."""
            try:
                lookahead = 24  # small number to pre-warm
                to_warm = []
                for lbl, p in pending_thumbs[:lookahead]:
                    to_warm.append(p)
                if not to_warm:
                    return
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as ex:
                    futures = [ex.submit(_process_image, None, p) for p in to_warm]
                    for f in futures:
                        try:
                            f.result()
                        except Exception:
                            pass
            except Exception:
                pass

        def _start_thumbs() -> None:
            if pending_thumbs or pending_folders:
                _prioritize_visible_items()
                # Warm cache for visible items in background
                threading.Thread(target=_warm_visible_cache, daemon=True).start()
                _load_thumbs()

        # --- Phase 1: batch-insert items, yielding to the event loop between
        #     batches so the UI stays responsive during large directories ---
        def _insert_batch(start: int, col: int, row: int) -> None:
            if self._nav_generation != current_nav or self._thumb_generation != current_gen:
                return  # stale — user navigated away, abort silently

            end = min(start + _BATCH, len(all_items))

            for i in range(start, end):
                item       = all_items[i]
                icon_frame, placeholder_label = self._create_icon_widget_fast(item, thumb_size)
                icon_frame.place(x=0, y=0)
                self.icon_buttons.append((icon_frame, item))
                self._gallery_flow_items.append(icon_frame)

                if placeholder_label is not None:
                    if item.is_dir():
                        pending_folders.append((placeholder_label, item))
                    else:
                        # Enqueue only an initial lookahead; defer the rest
                        if len(pending_thumbs) < LOOKAHEAD:
                            pending_thumbs.append((placeholder_label, item))
                        else:
                            deferred_thumbs.append((placeholder_label, item))

            if end < len(all_items):
                # More items remain — schedule the next batch and return so
                # the event loop can process any pending navigation/resize events.
                self.root.after(0, lambda: _insert_batch(end, col, row))
            else:
                # All items inserted — start thumbnails and lay out the initial flow.
                self._schedule_gallery_flow_layout()
                _start_thumbs()

                # After the first pass, enqueue deferred thumbs in chunks so
                # the workers don't get flooded. This runs only if there are
                # deferred items left.
                def _enqueue_deferred_chunk():
                    try:
                        if not deferred_thumbs:
                            return
                        # Move a small chunk into pending_thumbs
                        chunk = []
                        for _ in range(min(LOOKAHEAD, len(deferred_thumbs))):
                            chunk.append(deferred_thumbs.pop(0))
                        pending_thumbs.extend(chunk)
                        # Start another thumb-pass to process newly enqueued items
                        _start_thumbs()
                        # Schedule next chunk if still deferred
                        if deferred_thumbs:
                            self.root.after(300, _enqueue_deferred_chunk)
                    except Exception:
                        pass

                if deferred_thumbs:
                    self.root.after(300, _enqueue_deferred_chunk)

        _insert_batch(0, 0, 0)
        icons_ui_dt = (time.time() - icons_ui_start)
        if icons_ui_dt > 0.05:
            print(f"[UI] load_icons_view UI work (batch setup) took {icons_ui_dt:.3f}s")

    def _get_placeholder(self, size: int, rgb: tuple = (220, 220, 220)) -> ImageTk.PhotoImage:
        """Return a shared grey placeholder PhotoImage of *size*×*size*.

        The image is created once per (size, rgb) pair and cached for the
        lifetime of the app.  This eliminates the O(N) Image.new() +
        ImageTk.PhotoImage() calls that previously ran on the UI thread during
        gallery Phase 1 — the cost drops from N PIL operations to at most 2
        (one for image files, one for folders).
        """
        key = (size, rgb)
        ph  = self._placeholder_cache.get(key)
        if ph is None:
            ph = ImageTk.PhotoImage(Image.new('RGB', (size, size), rgb))
            self._placeholder_cache[key] = ph
        return ph

    def _create_icon_widget_fast(self, path, thumb_size=120):
        """Create an icon frame with an *emoji* placeholder immediately.

        Returns ``(frame, label_or_None)``.
        If the item is an image file, also returns the label so the caller can
        later swap in a real thumbnail.  Returns None as the label for folders
        and non-image files.
        """
        frame = ttk.Frame(self.icons_container)

        # Reserve a fixed-size holder for the image area to avoid layout
        # reflows when the placeholder is replaced by the real thumbnail.
        image_holder = ttk.Frame(frame, width=thumb_size, height=thumb_size)
        image_holder.pack()
        try:
            image_holder.pack_propagate(False)
        except Exception:
            pass

        is_image = (
            path.is_file()
            and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif',
                                         '.bmp', '.webp', '.tiff', '.ico', '.jfif'}
        )
        if is_image:
            # Check cache first — avoids the background wait for already-seen images
            cache_key = (str(path), thumb_size)
            with self._thumb_lock:
                cached = self.thumbnail_cache.get(cache_key)
            if cached is not None:
                icon_label = tk.Label(image_holder, image=cached, bg='white')
                icon_label.image = cached  # type: ignore
                icon_label.pack()
                self.image_references.append(cached)
                thumb_label = None  # no background work needed
            else:
                # Shared grey placeholder — created once and reused for every
                # uncached image (eliminates N × Image.new() on the UI thread).
                placeholder = self._get_placeholder(thumb_size, (220, 220, 220))
                icon_label  = tk.Label(image_holder, image=placeholder, bg='white')
                icon_label.image = placeholder  # type: ignore
                icon_label.pack()
                thumb_label = icon_label  # background thread will swap in real thumbnail
        elif path.is_dir():
            # Check collage cache first
            cache_key = ('folder_collage', str(path), thumb_size)
            with self._thumb_lock:
                cached = self.thumbnail_cache.get(cache_key)
            if cached is not None:
                icon_label = tk.Label(image_holder, image=cached, bg='white')
                icon_label.image = cached  # type: ignore
                icon_label.pack()
                self.image_references.append(cached)
                thumb_label = None
            else:
                # Shared slightly darker placeholder for folders
                placeholder = self._get_placeholder(thumb_size, (210, 210, 210))
                icon_label  = tk.Label(image_holder, image=placeholder, bg='white')
                icon_label.image = placeholder  # type: ignore
                icon_label.pack()
                thumb_label = icon_label  # background thread will replace with collage
        else:
            file_type_display, _ = self.get_file_type_from_extension(path)
            icon_text = file_type_display.split()[0] if file_type_display else '📄'
            icon_label = tk.Label(image_holder, text=icon_text, font=('Arial', 48), bg='white')
            icon_label.pack(expand=True)
            thumb_label = None

        # Name label — adaptive font so the full name fits ~one line, with at
        # most 1-2 overflow characters wrapping to a second row.  No truncation.
        name = path.name
        # Approximate: Arial char width ≈ 0.62× font size in pixels.
        # Solve for the largest font where the name fits in thumb_size width.
        char_w_factor = 0.62
        ideal_size = int((thumb_size - 4) / max(len(name) * char_w_factor, 1))
        font_size = max(7, min(13, ideal_size))
        name_label = tk.Label(
            frame, text=name,
            font=('Arial', font_size),
            bg='#e8e8e8',
            wraplength=thumb_size - 4,
            anchor='center',
            justify='center',
        )
        name_label.pack(fill='x')

        # Click bindings
        icon_label.bind('<Double-1>', lambda e: self.on_icon_double_click(path))
        name_label.bind('<Double-1>', lambda e: self.on_icon_double_click(path))
        icon_label.bind('<Button-1>', lambda e: self.on_icon_single_click(path))
        name_label.bind('<Button-1>', lambda e: self.on_icon_single_click(path))
        icon_label.bind('<Button-3>', lambda e: self._show_icon_context_menu(e, path))
        name_label.bind('<Button-3>', lambda e: self._show_icon_context_menu(e, path))
        frame.bind('<Button-3>', lambda e: self._show_icon_context_menu(e, path))

        # Drag-and-drop source bindings for the icon view
        for _w in (icon_label, name_label, frame):
            _w.bind('<ButtonPress-1>',
                    lambda e, p=path: self._on_icon_drag_start(e, p), add='+')
            _w.bind('<B1-Motion>',      self._on_icon_drag_motion, add='+')
            _w.bind('<ButtonRelease-1>', self._on_icon_drag_release, add='+')

        return frame, thumb_label
    
    # ------------------------------------------------------------------
    # Icon-view drag-and-drop
    # ------------------------------------------------------------------

    def _on_icon_drag_start(self, event, path: Path):
        self._drag_active = False
        self._drag_start_pos = (event.x_root, event.y_root)
        self._drag_source_path = path

    def _on_icon_drag_motion(self, event):
        if self._drag_source_path is None:
            return
        dx = abs(event.x_root - self._drag_start_pos[0])
        dy = abs(event.y_root - self._drag_start_pos[1])
        if dx >= self._DRAG_THRESHOLD or dy >= self._DRAG_THRESHOLD:
            self._drag_active = True
            try:
                self.icons_container.configure(cursor='fleur')
            except Exception:
                pass

    def _on_icon_drag_release(self, event):
        try:
            self.icons_container.configure(cursor='')
        except Exception:
            pass
        if not self._drag_active or self._drag_source_path is None:
            self._drag_active = False
            self._drag_source_path = None
            return
        src = self._drag_source_path
        self._drag_active = False
        self._drag_source_path = None

        dest_folder = self._get_folder_from_point(event.x_root, event.y_root)
        if dest_folder is None:
            return
        if dest_folder.resolve() == src.resolve():
            return
        if src.parent.resolve() == dest_folder.resolve():
            return
        if src.is_dir() and dest_folder.resolve() in src.resolve().parents:
            return

        def _do():
            target = dest_folder / src.name
            if target.exists():
                target = self._resolve_name_conflict(target)
            try:
                shutil.move(str(src), str(target))
            except Exception as exc:
                err = str(exc)
                self.root.after(0, lambda e=err: messagebox.showerror('Move Error', e))
                return
            self.root.after(0, lambda: (
                self.refresh(),
                self._refresh_inbox_count(),
                self.status_var.set(f'Moved {src.name} → {dest_folder.name}')
            ))

        threading.Thread(target=_do, daemon=True).start()

    def _get_folder_from_point(self, x_root: int, y_root: int):
        """Return the folder Path whose icon frame contains the abs screen point."""
        for frame, path in self.icon_buttons:
            try:
                fx = frame.winfo_rootx()
                fy = frame.winfo_rooty()
                fw = frame.winfo_width()
                fh = frame.winfo_height()
                if fx <= x_root <= fx + fw and fy <= y_root <= fy + fh:
                    return path if path.is_dir() else None
            except Exception:
                pass
        return None

    def on_icon_single_click(self, path):
        """Handle single-click on icon for selection"""
        if path.is_file():
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif']:
                self.on_image_selected(str(path))
    
    def on_icon_double_click(self, path):
        """Handle double-click on icon"""
        if path.is_dir():
            self.history.append(self.current_path)
            self.load_directory(path)
        elif path.is_file():
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif']:
                self.on_image_selected(str(path))
                self.open_file(path)
            else:
                self.open_file(path)
    
    def add_tree_item(self, path, item_type, category='file', skip_thumbnail=False):
        """Add an item to the tree view.

        When *skip_thumbnail* is True (the default for bulk loads) the item is
        inserted without a thumbnail so the call is cheap; callers can update
        the image later via tree.item().
        """
        try:
            # Get file stats
            stats = path.stat()

            # Format size
            size = self.format_size(stats.st_size) if path.is_file() else ""

            # Format creation and modification time
            from datetime import datetime
            created_time = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

            # Optionally attach a thumbnail that is already cached
            thumbnail = None
            if not skip_thumbnail and category == 'image' and path.is_file():
                cache_key = (str(path), 24)
                with self._thumb_lock:
                    thumbnail = self.thumbnail_cache.get(cache_key)
                if thumbnail is None:
                    thumbnail = self.create_thumbnail_for_tree(path, size=24)
                if thumbnail:
                    self.tree_thumbnails[str(path)] = thumbnail

            if thumbnail:
                item_id = self.tree.insert('', tk.END, text=path.name, image=thumbnail,
                           values=(item_type, size, created_time, mod_time),
                           tags=('folder' if path.is_dir() else 'file',))
            else:
                item_id = self.tree.insert('', tk.END, text=path.name,
                           values=(item_type, size, created_time, mod_time),
                           tags=('folder' if path.is_dir() else 'file',))
            return item_id

        except Exception:
            # If we can't get stats, still add the item
            return self.tree.insert('', tk.END, text=path.name,
                           values=(item_type, 'N/A', 'N/A', 'N/A'),
                           tags=('folder' if path.is_dir() else 'file',))
    
    # ------------------------------------------------------------------
    # Hidden-item helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_os_hidden(path: Path) -> bool:
        """Return True if *path* has the OS hidden attribute set.

        On Windows this checks FILE_ATTRIBUTE_HIDDEN (0x02).  On all other
        platforms a leading dot is the conventional indicator.
        """
        if platform.system() == 'Windows':
            try:
                import ctypes
                attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
                if attrs == -1:  # INVALID_FILE_ATTRIBUTES
                    return False
                return bool(attrs & 0x02)  # FILE_ATTRIBUTE_HIDDEN
            except Exception:
                return False
        return path.name.startswith('.')

    @staticmethod
    def _is_os_hidden_entry(entry: os.DirEntry) -> bool:  # type: ignore[type-arg]
        """Faster variant that works on a ``DirEntry`` from ``os.scandir()``.

        On Windows, ``DirEntry.stat()`` is populated from the directory read
        itself (no extra syscall), so ``st_file_attributes`` is free.
        On other platforms a leading dot is the conventional indicator.
        """
        if platform.system() == 'Windows':
            try:
                return bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x02)
            except Exception:
                return False
        return entry.name.startswith('.')

    def _put_thumbnail_cache(self, key, photo) -> None:
        """Insert *photo* into the bounded thumbnail cache.

        If the cache is at capacity the oldest entry is evicted and its
        PhotoImage is blanked so Tk releases the underlying pixel buffer.
        Must be called with ``_thumb_lock`` already held, or be the sole writer.
        """
        if key in self.thumbnail_cache:
            return
        if len(self.thumbnail_cache) >= self._THUMB_CACHE_MAX:
            oldest_key, oldest_photo = next(iter(self.thumbnail_cache.items()))
            del self.thumbnail_cache[oldest_key]
            try:
                oldest_photo.blank()
            except Exception:
                pass
        self.thumbnail_cache[key] = photo

    def _get_app_hidden_paths(self, folder: str) -> set:
        """Return absolute paths of images hidden inside the app DB for *folder*."""
        try:
            from data_layer import metadata_store as _ms
            return _ms.get_hidden_image_paths_in_folder(folder)
        except Exception:
            return set()

    def _toggle_app_hidden(self, image_path: str):
        """Toggle the app-level hidden flag for *image_path* and refresh."""
        from data_layer import metadata_store as _ms
        record = _ms.get_image_by_path(image_path)
        if record is None:
            # Register image first so we have an ID to store the flag against
            p = Path(image_path)
            record_id = _ms.add_image(image_path, p.name,
                                      p.stat().st_size if p.exists() else 0)
            currently_hidden = False
        else:
            record_id = record['id']
            currently_hidden = _ms.is_image_hidden(record_id)

        if record_id is not None:
            _ms.set_image_hidden(record_id, not currently_hidden)
            action = 'hidden' if not currently_hidden else 'revealed'
            self.status_var.set(f"Image {action}: {Path(image_path).name}")
            self.load_directory(self.current_path)

    def _show_tree_context_menu(self, event):
        """Display right-click context menu on tree items."""
        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return
        self.tree.selection_set(row_id)
        item_text = self.tree.item(row_id, 'text')
        item_path = self.current_path / item_text

        menu = self._build_file_context_menu(item_path)
        menu.tk_popup(event.x_root, event.y_root)

    def _show_icon_context_menu(self, event, path: Path):
        """Display right-click context menu for an icon widget."""
        menu = self._build_file_context_menu(path)
        menu.tk_popup(event.x_root, event.y_root)

    # ------------------------------------------------------------------
    # Database cleanup
    # ------------------------------------------------------------------

    def clean_orphan_databases_action(self):
        """Menu action: scan the root folder and remove stale .image_tags.db files."""
        from data_layer import metadata_store as _ms
        root = str(self.root_folder)
        if not messagebox.askyesno(
                "Clean Orphan Databases",
                f"This will delete all .image_tags.db files under:\n{root}\n"
                f"that are NOT the current active database, and also remove the"
                f" default data/image_metadata.db if it is redundant.\n\n"
                f"Continue?"):
            return
        removed = _ms.clean_orphan_databases(root)
        if removed:
            msg = f"Removed {len(removed)} orphan database(s):\n" + "\n".join(removed)
        else:
            msg = "No orphan databases found."
        messagebox.showinfo("Clean Orphan Databases", msg)
        self.status_var.set(f"Cleaned {len(removed)} orphan database(s)")

    # ------------------------------------------------------------------

    def format_size(self, size):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def on_tree_single_click(self, event):
        """Handle single-click on tree item for image selection.

        Rapid multiple events (e.g. double-click producing two single-click
        events, or tree rebuilds re-triggering selection) are debounced with a
        short delay so that only the last event triggers the (potentially slow)
        tag DB lookup.
        """
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        item_text = self.tree.item(item, 'text')
        new_path = self.current_path / item_text

        if new_path.is_file() and new_path.suffix.lower() in [
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif']:
            # Cancel any pending debounced call
            if self._tag_debounce_id is not None:
                self.root.after_cancel(self._tag_debounce_id)
            path_str = str(new_path)
            self._tag_debounce_id = self.root.after(
                150, lambda p=path_str: self._do_image_selected(p)
            )

    def _do_image_selected(self, image_path: str):
        """Debounced handler – called after the click debounce window expires."""
        self._tag_debounce_id = None
        self.on_image_selected(image_path)
    
    def on_tree_double_click(self, event):
        """Handle double-click on tree item"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_text = self.tree.item(item, 'text')
        new_path = self.current_path / item_text
        
        if new_path.is_dir():
            self.history.append(self.current_path)
            self.load_directory(new_path)
        elif new_path.is_file():
            # For images, select them and open in default photo app; for other files, open them
            if new_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif']:
                self.on_image_selected(str(new_path))
                self.open_file(new_path)
            else:
                self.open_file(new_path)
    
    def on_double_click(self, event):
        """Legacy handler for backward compatibility"""
        self.on_tree_double_click(event)
    
    def open_file(self, filepath):
        """Open file with its default application."""
        try:
            if platform.system() == 'Windows':
                os.startfile(filepath)
            elif platform.system() == 'Darwin':
                os.system(f'open "{filepath}"')
            else:
                os.system(f'xdg-open "{filepath}"')
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open file: {e}")

    def open_file_with_dialog(self, filepath):
        """Show the OS \"Open With\" dialog for *filepath*.

        On Windows this invokes the native \"How do you want to open this file?\"
        shell dialog via rundll32.  On macOS the same is achieved with
        ``open -a ...`` after the user picks an app.  On Linux xdg-open is
        used as a fallback (most desktop environments handle Open-With).
        """
        import subprocess
        try:
            if platform.system() == 'Windows':
                subprocess.run(
                    ['rundll32.exe', 'shell32.dll,OpenAs_RunDLL', str(filepath)],
                    check=False
                )
            elif platform.system() == 'Darwin':
                # macOS: ask the user to pick an app via a simple input dialog,
                # or fall back to the default handler if they cancel.
                app = simpledialog.askstring(
                    "Open With",
                    "Enter application name (e.g. Preview, TextEdit):\n"
                    "Leave blank to use the default app.",
                    parent=self.root
                )
                if app:
                    subprocess.run(['open', '-a', app, str(filepath)], check=False)
                else:
                    os.system(f'open "{filepath}"')
            else:  # Linux / other
                subprocess.run(['xdg-open', str(filepath)], check=False)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open \"Open With\" dialog: {e}")

    def _build_file_context_menu(self, item_path: Path) -> tk.Menu:
        """Build and return a context menu for *item_path* (used by both tree
        and icon views).  Includes clipboard (Cut/Copy/Paste), Rename, Delete.
        """
        menu = tk.Menu(self.root, tearoff=0)

        if item_path.is_file():
            is_image = item_path.suffix.lower() in {
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif', '.tiff', '.ico'
            }

            # --- Open / Open With ---
            menu.add_command(
                label='Open',
                command=lambda p=item_path: self.open_file(str(p))
            )
            menu.add_command(
                label='Open With…',
                command=lambda p=item_path: self.open_file_with_dialog(str(p))
            )
            menu.add_separator()

            # --- App-hidden toggle (images only) ---
            if is_image:
                from data_layer import metadata_store as _ms
                record = _ms.get_image_by_path(str(item_path))
                is_app_hidden = _ms.is_image_hidden(record['id']) if record else False
                hide_label = 'Reveal in This App' if is_app_hidden else 'Hide in This App'
                menu.add_command(
                    label=hide_label,
                    command=lambda p=str(item_path): self._toggle_app_hidden(p)
                )
                menu.add_separator()

        elif item_path.is_dir():
            menu.add_command(
                label='Open Folder',
                command=lambda p=item_path: self.open_file(str(p))
            )
            menu.add_separator()

        # Determine targets: use tree selection when item_path is in it,
        # otherwise fall back to just the right-clicked item.
        sel = self._get_selected_tree_paths()
        targets = sel if item_path in sel else [item_path]

        # --- Clipboard ---
        menu.add_command(
            label='Cut\t\t  Ctrl+X',
            command=lambda tt=targets: self._clip_cut_paths(tt)
        )
        menu.add_command(
            label='Copy\t\t  Ctrl+C',
            command=lambda tt=targets: self._clip_copy_paths(tt)
        )
        paste_state = 'normal' if self._clipboard_paths else 'disabled'
        menu.add_command(
            label='Paste\t\t  Ctrl+V',
            command=self._clip_paste,
            state=paste_state
        )
        menu.add_separator()

        # --- Rename ---
        menu.add_command(
            label='Rename\t\t  F2',
            command=lambda p=item_path: self._file_rename_path(p)
        )
        menu.add_separator()

        # --- Delete ---
        menu.add_command(
            label='Delete\t\t  Del',
            command=lambda tt=targets: self._file_delete_paths(tt)
        )
        menu.add_separator()

        menu.add_command(label='Refresh', command=self.refresh)
        return menu

    # ------------------------------------------------------------------
    # Clipboard helpers
    # ------------------------------------------------------------------

    def _get_selected_tree_paths(self) -> list:
        """Return the Path for every currently selected item in the tree view."""
        result = []
        for iid in self.tree.selection():
            name = self.tree.item(iid, 'text')
            p = self.current_path / name
            if p.exists():
                result.append(p)
        return result

    def _clip_cut_paths(self, paths: list):
        self._clipboard_paths = list(paths)
        self._clipboard_op = 'cut'
        self.status_var.set(f"Cut {len(paths)} item(s) — Ctrl+V to paste")

    def _clip_copy_paths(self, paths: list):
        self._clipboard_paths = list(paths)
        self._clipboard_op = 'copy'
        self.status_var.set(f"Copied {len(paths)} item(s) — Ctrl+V to paste")

    def _clip_cut(self):
        paths = self._get_selected_tree_paths()
        if paths:
            self._clip_cut_paths(paths)

    def _clip_copy(self):
        paths = self._get_selected_tree_paths()
        if paths:
            self._clip_copy_paths(paths)

    def _clip_paste(self, dest_folder: Path | None = None):
        """Paste clipboard contents into *dest_folder* (defaults to current path)."""
        if not self._clipboard_paths or not self._clipboard_op:
            return
        dest = dest_folder or self.current_path
        srcs = list(self._clipboard_paths)
        op   = self._clipboard_op

        verb = 'Copying' if op == 'copy' else 'Moving'
        total = len(srcs)
        dlg = MoveProgressDialog(self.root, f"{verb} {total} item(s)\u2026", total=total)

        def _do():
            moved = errors = 0
            for i, src in enumerate(srcs, 1):
                if not src.exists():
                    errors += 1
                    continue
                target = dest / src.name
                if target.exists() and target.resolve() != src.resolve():
                    target = self._resolve_name_conflict(target)
                try:
                    if op == 'copy':
                        if src.is_dir():
                            shutil.copytree(str(src), str(target))
                        else:
                            shutil.copy2(str(src), str(target))
                    else:
                        shutil.move(str(src), str(target))
                    moved += 1
                except Exception as exc:
                    errors += 1
                    err = str(exc)
                    self.root.after(0, lambda e=err: messagebox.showerror('Paste Error', e))
                fname = src.name
                self.root.after(0, lambda n=i, f=fname, m=moved, e=errors:
                                dlg.update(f"\u2192 {dest.name}/{f}", n, e))
            if op == 'cut':
                self._clipboard_paths.clear()
                self._clipboard_op = None
            self.root.after(0, lambda m=moved, e=errors, d=dest.name: (
                dlg.mark_done(m, e),
                self.refresh(),
                self._refresh_inbox_count(),
                self.status_var.set(f'Pasted {m} item(s) into {d}')
            ))

        threading.Thread(target=_do, daemon=True).start()

    def _resolve_name_conflict(self, target: Path) -> Path:
        """Return a non-colliding rename of *target* by appending (1), (2), …"""
        stem, ext = target.stem, target.suffix
        counter = 1
        while True:
            candidate = target.parent / f'{stem} ({counter}){ext}'
            if not candidate.exists():
                return candidate
            counter += 1

    def _file_rename_path(self, src: Path):
        new_name = simpledialog.askstring(
            'Rename', f'New name for "{src.name}":', initialvalue=src.name, parent=self.root)
        if not new_name or new_name == src.name:
            return
        dest = src.parent / new_name
        if dest.exists():
            messagebox.showerror('Rename', f'"{new_name}" already exists.')
            return
        try:
            src.rename(dest)
            self.refresh()
            self.status_var.set(f'Renamed → {new_name}')
        except Exception as e:
            messagebox.showerror('Rename Error', str(e))

    def _file_rename(self):
        paths = self._get_selected_tree_paths()
        if paths:
            self._file_rename_path(paths[0])

    def _file_delete_paths(self, paths: list):
        if not paths:
            return
        names = '\n'.join(p.name for p in paths[:6])
        if len(paths) > 6:
            names += f'\n…and {len(paths) - 6} more'
        if not messagebox.askyesno('Delete', f'Permanently delete?\n\n{names}', icon='warning'):
            return
        affected_folders: set[Path] = set()
        for p in paths:
            try:
                affected_folders.add(p.parent)
                if p.is_dir():
                    shutil.rmtree(str(p))
                else:
                    p.unlink()
            except Exception as e:
                messagebox.showerror('Delete Error', str(e))
        for folder_path in sorted(affected_folders, key=lambda item: str(item)):
            self._rebuild_folder_preview_async(folder_path)
        self.refresh()

    def _file_delete(self):
        paths = self._get_selected_tree_paths()
        if paths:
            self._file_delete_paths(paths)

    # ------------------------------------------------------------------
    # Internal drag-and-drop (Treeview rows)
    # ------------------------------------------------------------------

    _DRAG_THRESHOLD = 6   # pixels before a drag is recognised

    def _on_tree_drag_start(self, event):
        self._drag_active = False
        self._drag_start_pos = (event.x_root, event.y_root)
        self._drag_source_items = []
        for iid in self.tree.selection():
            name = self.tree.item(iid, 'text')
            p = self.current_path / name
            if p.exists():
                self._drag_source_items.append((iid, p))

    def _on_tree_drag_motion(self, event):
        if not self._drag_source_items:
            return
        dx = abs(event.x_root - self._drag_start_pos[0])
        dy = abs(event.y_root - self._drag_start_pos[1])
        if dx < self._DRAG_THRESHOLD and dy < self._DRAG_THRESHOLD:
            return
        self._drag_active = True
        self.tree.configure(cursor='fleur')

        target_id = self.tree.identify_row(event.y)
        if target_id == self._drag_highlight_id:
            return
        # Clear old highlight
        if self._drag_highlight_id:
            try:
                self.tree.item(self._drag_highlight_id, tags=())
            except Exception:
                pass
            self._drag_highlight_id = None

        if target_id:
            item_name = self.tree.item(target_id, 'text')
            target_path = self.current_path / item_name
            if target_path.is_dir():
                self.tree.tag_configure('_dnd_target', background='#b0d4f8')
                self.tree.item(target_id, tags=('_dnd_target',))
                self._drag_highlight_id = target_id

    def _on_tree_drag_release(self, event):
        self.tree.configure(cursor='')
        if self._drag_highlight_id:
            try:
                self.tree.item(self._drag_highlight_id, tags=())
            except Exception:
                pass
            self._drag_highlight_id = None

        if not self._drag_active or not self._drag_source_items:
            self._drag_active = False
            return
        self._drag_active = False

        target_id = self.tree.identify_row(event.y)
        if not target_id:
            return
        item_name = self.tree.item(target_id, 'text')
        dest_folder = self.current_path / item_name
        if not dest_folder.is_dir():
            return

        sources = [
            p for _, p in self._drag_source_items
            if (p.resolve() != dest_folder.resolve()
                and dest_folder.resolve() not in p.resolve().parents)
        ]
        if not sources:
            return

        total = len(sources)
        dlg = MoveProgressDialog(self.root, f"Moving {total} item(s)\u2026", total=total)

        def _do_move():
            moved = errors = 0
            for i, src in enumerate(sources, 1):
                target = dest_folder / src.name
                if target.exists():
                    target = self._resolve_name_conflict(target)
                try:
                    shutil.move(str(src), str(target))
                    moved += 1
                except Exception as exc:
                    errors += 1
                    err = str(exc)
                    self.root.after(0, lambda e=err: messagebox.showerror('Move Error', e))
                fname = src.name
                self.root.after(0, lambda n=i, f=fname, m=moved, e=errors:
                                dlg.update(f"\u2192 {dest_folder.name}/{f}", n, e))
            self.root.after(0, lambda m=moved, e=errors, d=dest_folder.name: (
                dlg.mark_done(m, e),
                self.refresh(),
                self._refresh_inbox_count(),
                self.status_var.set(f'Moved {m} item(s) \u2192 {d}')
            ))

        threading.Thread(target=_do_move, daemon=True).start()

    # ------------------------------------------------------------------
    # OS-level drop target (requires tkinterdnd2)
    # ------------------------------------------------------------------

    def _setup_os_drop_target(self):
        """Register tree and canvas as OS drop targets when tkinterdnd2 is available."""
        if not _TKDND_AVAILABLE:
            return
        for widget in (self.tree, self.canvas):
            try:
                w: _Any = widget
                w.drop_target_register(DND_FILES)
                w.dnd_bind('<<DropEnter>>', self._on_os_drop_enter)
                w.dnd_bind('<<DropLeave>>', self._on_os_drop_leave)
                w.dnd_bind('<<Drop>>',      self._on_os_drop)
            except Exception:
                pass

    def _on_os_drop_enter(self, event):
        try:
            event.widget.configure(background='#b0d4f8')
        except Exception:
            pass

    def _on_os_drop_leave(self, event):
        try:
            event.widget.configure(background='')
        except Exception:
            pass

    def _on_os_drop(self, event):
        try:
            event.widget.configure(background='')
        except Exception:
            pass
        import re
        raw   = event.data
        parts = re.findall(r'\{([^}]+)\}|(\S+)', raw)
        paths = [p[0] or p[1] for p in parts if (p[0] or p[1])]
        dest  = self.current_path
        if not paths:
            return

        total = len(paths)
        dlg = MoveProgressDialog(self.root, f"Dropping {total} item(s) into {dest.name}\u2026",
                                 total=total)

        def _do():
            moved = errors = 0
            for i, src_str in enumerate(paths, 1):
                src = Path(src_str)
                if not src.exists() or src.parent.resolve() == dest.resolve():
                    errors += 1
                    self.root.after(0, lambda n=i, e=errors: dlg.update("(skipped)", n, e))
                    continue
                target = dest / src.name
                if target.exists():
                    target = self._resolve_name_conflict(target)
                try:
                    shutil.move(str(src), str(target))
                    moved += 1
                except Exception as exc:
                    errors += 1
                    err = str(exc)
                    self.root.after(0, lambda e=err: messagebox.showerror('Drop Error', e))
                fname = src.name
                self.root.after(0, lambda n=i, f=fname, m=moved, e=errors:
                                dlg.update(f"\u2192 {dest.name}/{f}", n, e))
            self.root.after(0, lambda m=moved, e=errors, d=dest.name: (
                dlg.mark_done(m, e),
                self.refresh(),
                self._refresh_inbox_count(),
                self.status_var.set(f'Dropped {m} item(s) into {d}')
            ))

        threading.Thread(target=_do, daemon=True).start()

    def go_back(self):
        """Go to previous directory"""
        if self.history:
            previous_path = self.history.pop()
            self._nav_forward_stack.append(self.current_path)
            self.load_directory(previous_path)
    
    def go_up(self):
        """Go to parent directory"""
        parent = self.current_path.parent
        if parent != self.current_path:  # Not at root
            self.history.append(self.current_path)
            self._nav_forward_stack.clear()
            self.load_directory(parent)
    
    def go_home(self):
        """Go to home directory"""
        self.history.append(self.current_path)
        self._nav_forward_stack.clear()
        self.load_directory(Path.home())
    
    def refresh(self):
        """Refresh current directory"""
        self.load_directory(self.current_path)
    
    def navigate_to_path(self):
        """Navigate to path entered in address bar"""
        path_str = self.path_var.get()
        try:
            path = Path(path_str)
            if path.exists() and path.is_dir():
                self.history.append(self.current_path)
                self._nav_forward_stack.clear()
                self.load_directory(path)
            else:
                messagebox.showerror("Error", f"Invalid path: {path_str}")
                self.path_var.set(str(self.current_path))
        except Exception as e:
            messagebox.showerror("Error", f"Invalid path: {e}")
            self.path_var.set(str(self.current_path))

    def set_root_folder_dialog(self):
        """Open a settings dialog to configure the root folder, tag root, and DB location."""
        from tkinter import filedialog

        win = Toplevel(self.root)
        win.title("Root Folder Settings")
        win.geometry("600x320")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        font_label  = ('Arial', 12, 'bold')
        font_normal = ('Arial', 12)

        def _browse(var):
            folder = filedialog.askdirectory(title="Select Folder",
                                             initialdir=var.get() or str(self.root_folder))
            if folder:
                var.set(folder)

        # --- Root folder row ---
        root_var = tk.StringVar(value=str(self.root_folder))
        ttk.Label(win, text="Root folder (database lives here):", font=font_label).pack(anchor='w', padx=10, pady=(12, 0))
        rf = ttk.Frame(win); rf.pack(fill=tk.X, padx=10, pady=2)
        ttk.Entry(rf, textvariable=root_var, font=font_normal).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(rf, text="Browse…", command=lambda: _browse(root_var)).pack(side=tk.LEFT)

        # --- Tag root folder row ---
        tag_root_var = tk.StringVar(value=config_manager.app_config.tag_root_folder or str(self.root_folder))
        ttk.Label(win, text="Tag root folder (folder-tag extraction starts here):", font=font_label).pack(anchor='w', padx=10, pady=(10, 0))
        tr = ttk.Frame(win); tr.pack(fill=tk.X, padx=10, pady=2)
        ttk.Entry(tr, textvariable=tag_root_var, font=font_normal).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(tr, text="Browse…", command=lambda: _browse(tag_root_var)).pack(side=tk.LEFT)

        # --- Auto-folder-tags checkbox ---
        auto_var = tk.BooleanVar(value=config_manager.app_config.auto_folder_tags)
        ttk.Checkbutton(win, text="Automatically add folder-path tags when scanning images",
                        variable=auto_var).pack(anchor='w', padx=10, pady=(8, 0))

        # --- Buttons ---
        btn_frame = ttk.Frame(win)
        btn_frame.pack(side=tk.BOTTOM, pady=12)

        def _apply():
            new_root     = root_var.get().strip()
            new_tag_root = tag_root_var.get().strip()
            if not new_root or not Path(new_root).exists():
                messagebox.showerror("Error", f"Root folder does not exist:\n{new_root}", parent=win)
                return

            # Apply root folder change
            self.root_folder = Path(new_root)
            config_manager.app_config.tag_root_folder  = new_tag_root
            config_manager.app_config.auto_folder_tags = auto_var.get()
            config_manager.save_config()

            # Reinitialise the controller with the new root
            self.controller.set_root_folder(new_root)
            self._ensure_inbox_exists()  # create _INBOX under the new root

            # Re-point thumbnail cache and LMDB to the new root so previews
            # are read/written in the correct location
            self.disk_thumbnail_dir = self.root_folder / '.thumbs'
            try:
                self.disk_thumbnail_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                if self._lmdb_env is not None:
                    self._lmdb_env.close()
            except Exception:
                pass
            try:
                if lmdb is not None:
                    self._lmdb_env = lmdb.open(
                        str(self.root_folder / '.thumbs.lmdb'),
                        map_size=1024**3, max_dbs=1)
                else:
                    self._lmdb_env = None
            except Exception:
                self._lmdb_env = None

            # Reload app state from the new root
            self._app_state_path = self.root_folder / '.app_state.json'
            self._load_app_state()

            messagebox.showinfo("Applied",
                                f"Root folder set to:\n{new_root}\n\n"
                                f"Tag root:\n{new_tag_root}\n\n"
                                "Database has been moved/created in the root folder.",
                                parent=win)
            win.destroy()
            self.status_var.set(f"Root folder: {new_root}")

        ttk.Button(btn_frame, text="Apply", command=_apply, width=12).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy, width=12).pack(side=tk.LEFT, padx=6)
    
    def setup_tag_sidebar(self, parent):
        """Setup the compact tag management sidebar with a collapse/expand toggle."""
        # ── Wrapper (always packed RIGHT) ──────────────────────────────────
        # The toggle strip sits on the left edge of the wrapper; the content
        # panel sits to its right.  Collapsing hides the content panel so the
        # wrapper shrinks to just the strip width.
        sidebar_wrapper = ttk.Frame(parent)
        sidebar_wrapper.pack(side=tk.RIGHT, fill=tk.Y)

        # Toggle strip — stays visible when collapsed
        self._sidebar_expanded = tk.BooleanVar(value=True)
        self._sidebar_toggle_btn = ttk.Button(
            sidebar_wrapper, text="◀", width=2,
            command=self._toggle_tag_sidebar,
        )
        self._sidebar_toggle_btn.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))

        # Content panel
        tag_sidebar = ttk.Frame(sidebar_wrapper, width=260)
        tag_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        tag_sidebar.pack_propagate(False)
        self.tag_sidebar = tag_sidebar
        self._tag_sidebar_content = tag_sidebar  # reference for toggle

        # ---- Header (fixed, outside scroll area) ----
        ttk.Label(tag_sidebar, text="Tags", font=('Arial', 11, 'bold')).pack(pady=(3, 0))
        self.selected_image_label = ttk.Label(
            tag_sidebar, text="No image selected",
            wraplength=250, font=('Arial', 9), foreground='gray'
        )
        self.selected_image_label.pack(pady=(0, 2))
        ttk.Separator(tag_sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ---- Scrollable body ----
        tag_canvas = Canvas(tag_sidebar, highlightthickness=0)
        tag_scrollbar = ttk.Scrollbar(tag_sidebar, orient=tk.VERTICAL, command=tag_canvas.yview)
        tag_scrollable_frame = ttk.Frame(tag_canvas)

        tag_scrollable_frame.bind(
            "<Configure>",
            lambda e: tag_canvas.configure(scrollregion=tag_canvas.bbox("all"))
        )
        tag_canvas_window = tag_canvas.create_window((0, 0), window=tag_scrollable_frame, anchor="nw")
        tag_canvas.configure(yscrollcommand=tag_scrollbar.set)

        def _on_tag_canvas_resize(event):
            tag_canvas.itemconfig(tag_canvas_window, width=event.width)
        tag_canvas.bind("<Configure>", _on_tag_canvas_resize)

        def _on_tag_wheel(event):
            tag_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        tag_canvas.bind("<MouseWheel>", _on_tag_wheel)
        tag_scrollable_frame.bind("<MouseWheel>", _on_tag_wheel)

        def _bind_wheel_recursive(widget):
            widget.bind("<MouseWheel>", _on_tag_wheel, add="+")
            for child in widget.winfo_children():
                _bind_wheel_recursive(child)
        tag_sidebar.after(200, lambda: _bind_wheel_recursive(tag_scrollable_frame))

        tag_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tag_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sf  = tag_scrollable_frame   # shorthand
        _p: dict = {'padx': 4, 'pady': 1}

        # ---- Current Tags ----
        ttk.Label(sf, text="Current Tags:", font=('Arial', 10, 'bold')).pack(anchor='w', **_p)
        tags_frame = ttk.Frame(sf)
        tags_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=1)
        self.tags_listbox = tk.Listbox(tags_frame, selectmode=tk.SINGLE, height=4, font=('Arial', 10))
        _tsb = ttk.Scrollbar(tags_frame, orient=tk.VERTICAL, command=self.tags_listbox.yview)
        self.tags_listbox.configure(yscrollcommand=_tsb.set)
        self.tags_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _tsb.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(sf, text="✕ Remove Tag", command=self.remove_selected_tag).pack(fill=tk.X, padx=4, pady=1)

        ttk.Separator(sf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # ---- Add Tag ----
        ttk.Label(sf, text="Add Tag:", font=('Arial', 10, 'bold')).pack(anchor='w', **_p)
        add_row = ttk.Frame(sf)
        add_row.pack(fill=tk.X, padx=4, pady=1)
        self.new_tag_entry = ttk.Entry(add_row, font=('Arial', 10))
        self.new_tag_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(add_row, text="Add", command=self.add_manual_tag, width=5).pack(side=tk.LEFT)
        self.new_tag_entry.bind('<Return>', lambda e: self.add_manual_tag())

        ttk.Separator(sf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # ---- Quick Actions ----
        ttk.Label(sf, text="Quick Actions:", font=('Arial', 10, 'bold')).pack(anchor='w', **_p)
        qa_row = ttk.Frame(sf)
        qa_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(qa_row, text="🗂 Folder Tags",
                   command=self.add_folder_tags_to_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        ttk.Button(qa_row, text="⚙ Root…",
                   command=self.set_root_folder_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        ttk.Separator(sf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # ---- Predicted Tags ----
        ttk.Label(sf, text="Predicted Tags:", font=('Arial', 10, 'bold')).pack(anchor='w', **_p)
        predictions_frame = ttk.Frame(sf)
        predictions_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=1)
        self.predictions_listbox = tk.Listbox(predictions_frame, selectmode=tk.SINGLE, height=3, font=('Arial', 10))
        _psb = ttk.Scrollbar(predictions_frame, orient=tk.VERTICAL, command=self.predictions_listbox.yview)
        self.predictions_listbox.configure(yscrollcommand=_psb.set)
        self.predictions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _psb.pack(side=tk.RIGHT, fill=tk.Y)
        pred_row = ttk.Frame(sf)
        pred_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(pred_row, text="Predict",
                   command=self.predict_tags_for_selected).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        ttk.Button(pred_row, text="Apply Sel.",
                   command=self.apply_selected_prediction).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))

        ttk.Separator(sf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)

        # ---- Inbox ----
        inbox_hdr = ttk.Frame(sf)
        inbox_hdr.pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(inbox_hdr, text="📥 Inbox:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.inbox_count_label = ttk.Label(inbox_hdr, text="–", font=('Arial', 10), foreground='gray')
        self.inbox_count_label.pack(side=tk.LEFT, padx=(4, 0))
        inbox_row = ttk.Frame(sf)
        inbox_row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(inbox_row, text="⚡ Tag Inbox",
                   command=self.process_inbox_tags).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        ttk.Button(inbox_row, text="📂 Move Tagged",
                   command=self.move_inbox_to_folders).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))
        ttk.Button(sf, text="✏ Tag & Fine-Tune",
                   command=self.open_inbox_training).pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(sf, text="🗂 Open Inbox Folder",
                   command=self.navigate_to_inbox).pack(fill=tk.X, padx=4, pady=1)
        ttk.Label(sf, text="Auto-tag & move images to matching\nsubfolders based on predicted tags.",
                  font=('Arial', 9), foreground='gray', justify='left').pack(anchor='w', padx=4, pady=(1, 4))

        # Refresh inbox count once sidebar is realised
        tag_sidebar.after(600, self._refresh_inbox_count)
    
    # ------------------------------------------------------------------
    # Inbox helpers
    # ------------------------------------------------------------------

    def _toggle_tag_sidebar(self) -> None:
        """Collapse or expand the right-side tag sidebar."""
        if self._sidebar_expanded.get():
            # Collapse — hide content, shrink to the toggle strip
            self._tag_sidebar_content.pack_forget()
            self._sidebar_toggle_btn.config(text="▶")
            self._sidebar_expanded.set(False)
        else:
            # Expand — restore content panel to the right of the strip
            self._tag_sidebar_content.pack(side=tk.LEFT, fill=tk.Y)
            self._sidebar_toggle_btn.config(text="◀")
            self._sidebar_expanded.set(True)

    def _ensure_inbox_exists(self) -> Path:
        """Create the _INBOX folder inside the root folder if it does not exist."""
        inbox = self.root_folder / _INBOX_FOLDER
        try:
            inbox.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return inbox

    def navigate_to_inbox(self):
        """Navigate the file browser directly to the _INBOX folder."""
        inbox = self._ensure_inbox_exists()
        self.history.append(self.current_path)
        self.load_directory(inbox)

    def open_inbox_training(self):
        """Open the Inbox Tag & Fine-Tune dialog."""
        try:
            inbox = self._ensure_inbox_exists()
            # Check inbox has images
            try:
                images = [
                    f for f in inbox.iterdir()
                    if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS
                    and f.name != _PREVIEW_FILENAME
                ]
            except Exception:
                images = []
            if not images:
                messagebox.showinfo(
                    'Inbox Empty',
                    'The inbox folder is empty.\nDrop images into:\n' + str(inbox))
                return

            known_tags = self.controller.get_known_tags()
            if not known_tags:
                messagebox.showwarning(
                    'No Model',
                    'No trained model is loaded yet.\n\n'
                    'Train the model first (bootstrap from folders or '
                    'train from database), then use Tag & Fine-Tune.')
                return

            InboxTrainingDialog(self.root, inbox, self.controller, known_tags)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror(
                'Tag & Fine-Tune Error',
                f'Could not open the dialog:\n{exc}')

    def _refresh_inbox_count(self):
        """Update the inbox image-count label."""
        if not hasattr(self, 'inbox_count_label'):
            return
        inbox = self.root_folder / _INBOX_FOLDER
        if not inbox.exists():
            self.inbox_count_label.config(text='(none)')
            return
        try:
            count = sum(1 for f in inbox.iterdir()
                        if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS
                        and f.name != _PREVIEW_FILENAME)
            self.inbox_count_label.config(text=f'{count} image(s)')
        except Exception:
            self.inbox_count_label.config(text='?')

    def process_inbox_tags(self):
        """Auto-tag all images in the _INBOX folder using the current ML model."""
        inbox = self._ensure_inbox_exists()
        images = [str(f) for f in sorted(inbox.iterdir())
                  if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS
                  and f.name != _PREVIEW_FILENAME]
        if not images:
            messagebox.showinfo("Inbox", "Inbox is empty.\nDrop images into:\n" + str(inbox))
            return
        if not messagebox.askyesno("Tag Inbox",
                f"Predict and apply tags to {len(images)} inbox image(s)?\n"
                f"Requires a trained ML model."):
            return

        def _run():
            ok = 0
            for img_path in images:
                try:
                    preds = self.controller.tagging_engine.predict_tags_for_image(img_path)
                    for pred in preds[:3]:   # apply top-3 predictions
                        self.controller.tagging_engine.apply_manual_tag(
                            img_path, pred['tag_name'], is_ground_truth=False
                        )
                    ok += 1
                except Exception:
                    pass
            self.root.after(0, lambda n=ok: (
                self._refresh_inbox_count(),
                self.status_var.set(f"Inbox: tagged {n}/{len(images)} image(s).")
            ))

        threading.Thread(target=_run, daemon=True).start()
        self.status_var.set("Tagging inbox images…")

    def move_inbox_to_folders(self):
        """Move inbox images that have tags into the best-matching root subfolder."""
        inbox = self._ensure_inbox_exists()
        images = [f for f in sorted(inbox.iterdir())
                  if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS
                  and f.name != _PREVIEW_FILENAME]
        if not images:
            messagebox.showinfo("Inbox", "Inbox is empty.")
            return

        # Build lowercase-name → Path map for root subfolders (fast, OK on UI thread)
        try:
            subfolders = {
                sf.name.lower(): sf
                for sf in self.root_folder.iterdir()
                if sf.is_dir()
                and sf.name != _INBOX_FOLDER
                and not self._is_os_hidden(sf)
            }
        except Exception as exc:
            messagebox.showerror("Error", f"Cannot list root subfolders:\n{exc}")
            return

        total = len(images)
        dlg = MoveProgressDialog(self.root, f"Moving {total} inbox image(s)…", total=total)

        def _run():
            moved = skipped = 0
            for i, img_path in enumerate(images, 1):
                tags = self.controller.get_image_tags(str(img_path))
                tags_sorted = sorted(tags, key=lambda t: t.get('confidence', 0), reverse=True)

                dest_folder = None
                for tag in tags_sorted:
                    tl = tag['tag_name'].lower()
                    if tl in subfolders:
                        dest_folder = subfolders[tl]
                        break
                    for sf_name, sf_path in subfolders.items():
                        if tl in sf_name or sf_name in tl:
                            dest_folder = sf_path
                            break
                    if dest_folder:
                        break

                if dest_folder is None:
                    skipped += 1
                    self.root.after(0, lambda n=i, f=img_path.name, s=skipped:
                                    dlg.update(f"Skipping {f} (no matching folder)", n, s))
                    continue

                dest_file = dest_folder / img_path.name
                if dest_file.exists():
                    dest_file = dest_folder / f"{img_path.stem}_{int(time.time())}{img_path.suffix}"
                try:
                    shutil.move(str(img_path), str(dest_file))
                    moved += 1
                except Exception:
                    skipped += 1

                df_name = dest_folder.name if dest_folder else '?'
                self.root.after(0, lambda n=i, f=img_path.name, d=df_name, m=moved, s=skipped:
                                dlg.update(f"→ {d}/{f}", n, s))

            self.root.after(0, lambda m=moved, s=skipped: (
                dlg.mark_done(m, s),
                self._refresh_inbox_count(),
                self.load_directory(self.current_path),
                self.status_var.set(f"Inbox: moved {m}, skipped {s}.")
            ))

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------

    def _startup_sequence(self) -> None:
        """Run startup tasks after the UI event loop is alive.

        Shows a blocking splash screen while the ML system initialises and the
        initial directory listing loads.  Both tasks run concurrently; the
        splash is dismissed as soon as **both** have finished (or after a 30 s
        safety timeout in case something hangs).
        """
        splash = SplashScreen(self.root)

        _done: dict = {'ml': False, 'dir': False}

        def _check_done() -> None:
            if all(_done.values()):
                splash.set_status('Ready!')
                # Brief "Ready" flash, then close splash and reveal the path
                self.root.after(220, splash.dismiss)
                self.root.after(540, lambda: self._nav_tree_reveal_path(self.current_path))

        def _ml_done() -> None:
            _done['ml'] = True
            _check_done()

        def _dir_done() -> None:
            _done['dir'] = True
            _check_done()

        # Safety: always dismiss after 30 s (handles model-load hangs, etc.)
        self.root.after(30_000, splash.dismiss)

        # Hook into load_directory's _show callback
        self._startup_dir_done_callback = _dir_done

        splash.set_status('Loading directory…')
        self.load_directory(self.current_path)

        self.initialize_ml_system(
            status_callback=splash.set_status,
            on_done=_ml_done,
        )

        try:
            self._restore_ui_state()
        except Exception:
            pass

    def _sync_search_availability(self) -> None:
        """Enable or disable advanced search controls based on model readiness."""
        model_ready = bool(getattr(self.controller.tagging_engine, 'model_loaded', False))
        widgets = [
            getattr(self, 'search_entry', None),
            getattr(self, '_sort_dir_btn', None),
            getattr(self, 'search_clear_btn', None),
            getattr(self, '_all_tags_listbox', None),
        ]
        for widget in widgets:
            try:
                if widget is not None:
                    widget.configure(state=tk.NORMAL if model_ready else tk.DISABLED)
            except Exception:
                pass

    def initialize_ml_system(self, status_callback=None, on_done=None):
        """Initialize the ML system in a background thread.

        *status_callback(msg)* — called (via root.after) with human-readable
        step descriptions while work is in progress.
        *on_done()* — called (via root.after) once the thread finishes,
        regardless of success or failure.
        """
        def _status(msg: str) -> None:
            if status_callback:
                try:
                    self.root.after(0, lambda m=msg: status_callback(m))
                except Exception:
                    pass

        def init_thread():
            try:
                _status('Setting up working directory…')
                self.controller.set_root_folder(str(self.root_folder))
                self._ensure_inbox_exists()

                _status('Connecting to database…')
                self.controller.initialize(str(self.current_path))

                try:
                    self.root.after(0, self._refresh_trained_models)
                except Exception:
                    pass

                _status('Checking for saved model…')
                try:
                    if model_storage.model_exists('latest'):
                        _status('Loading model weights…')
                    self.controller.try_auto_load_model()
                except Exception:
                    pass

                if self.controller.tagging_engine.model_loaded:
                    _status('✓ Model loaded — ready')
                    self.root.after(0, lambda: self.status_var.set(
                        "✓ ML system ready — model loaded"))
                    self.root.after(0, lambda: self.model_status_label.config(
                        text=f"✓ Model loaded ({config_manager.app_config.model_type})",
                        foreground='green'))
                else:
                    _status('⚠ No saved model found')
                    self.root.after(0, lambda: self.status_var.set(
                        "⚠ ML system ready — no model loaded (train a model first)"))
                    self.root.after(0, lambda: self.model_status_label.config(
                        text="⚠ No model loaded", foreground='orange'))

                self.root.after(0, self._sync_search_availability)

            except Exception as e:
                _status(f'Error: {str(e)[:80]}')
                self.root.after(0, lambda err=str(e)[:50]:
                    self.status_var.set(f"ML initialization error: {err}"))
                self.root.after(0, lambda: self.model_status_label.config(
                    text="✗ Initialization error", foreground='red'))

            finally:
                if on_done:
                    try:
                        self.root.after(0, on_done)
                    except Exception:
                        pass

        threading.Thread(target=init_thread, daemon=True).start()
    
    def on_image_selected(self, image_path: str):
        """Handle image selection for tagging (fast path on UI thread, DB work in background)"""
        self.selected_image_path = image_path
        path_obj = Path(image_path)
        self.selected_image_label.config(text=f"Selected: .../{path_obj.name}")
        
        # Load current tags in background so DB query doesn't freeze UI
        self._load_image_tags_async()
        
        # Auto-predict only if supervised mode is enabled AND the predictions
        # pane is visible (prevents background inference when the pane is hidden)
        try:
            preds_visible = getattr(self, 'predictions_listbox', None) is not None and self.predictions_listbox.winfo_ismapped()
        except Exception:
            preds_visible = False
        if config_manager.app_config.supervised_mode and preds_visible:
            self._predict_tags_async()
    
    def _load_image_tags_async(self):
        """Load tags for selected image in background thread"""
        def _worker():
            if self.selected_image_path is None:
                return
            try:
                tags = self.controller.get_image_tags(self.selected_image_path)
                self.root.after(0, lambda t=tags: self._update_tags_listbox(t))
            except Exception as e:
                print(f"[Error] loading tags: {e}")
                self.root.after(0, lambda: self._update_tags_listbox([]))
        
        threading.Thread(target=_worker, daemon=True).start()
    
    def _update_tags_listbox(self, tags):
        """Update UI with tags (run on UI thread via root.after)"""
        self.tags_listbox.delete(0, tk.END)
        self.selected_image_tags = tags
        for tag in tags:
            confidence = tag.get('confidence', 1.0)
            source = tag.get('source', 'unknown')
            display_text = f"{tag['tag_name']} ({confidence:.2f}) [{source}]"
            self.tags_listbox.insert(tk.END, display_text)
    
    def _predict_tags_async(self):
        """Predict tags in background so DB/model operations don't freeze UI"""
        if self.selected_image_path is None:
            return
        
        selected_path = self.selected_image_path  # Capture in local var
        
        def _worker():
            try:
                self.root.after(0, lambda: self.predictions_listbox.delete(0, tk.END))
                self.root.after(0, lambda: self.predictions_listbox.insert(tk.END, "Predicting..."))
                
                predictions = self.controller.predict_tags_for_image(selected_path)
                
                self.root.after(0, lambda p=predictions: self._update_predictions_listbox(p))
            except Exception as e:
                err_msg = str(e)[:200]
                print(f"[Error] predicting tags: {e}")
                self.root.after(0, lambda em=err_msg: (
                    self.predictions_listbox.delete(0, tk.END),
                    self.predictions_listbox.insert(tk.END, f"Error: {em}")
                ))
        
        threading.Thread(target=_worker, daemon=True).start()
    
    def _update_predictions_listbox(self, predictions):
        """Update predictions UI (run on UI thread)"""
        self.predictions_listbox.delete(0, tk.END)
        if not predictions:
            self.predictions_listbox.insert(tk.END, "(No predictions)")
            return
        for pred in predictions:
            conf = pred.get('confidence', 0.0)
            display_text = f"{pred['tag_name']} ({conf:.2f})"
            self.predictions_listbox.insert(tk.END, display_text)

    def on_load_model(self):
        """User-triggered model load button handler (runs in background)."""
        def _worker():
            try:
                self.root.after(0, lambda: self.model_status_label.config(text='Loading model...', foreground='blue'))
                model_name = ''
                try:
                    model_name = str(self.trained_model_var.get() or '').strip()
                except Exception:
                    model_name = ''
                if not model_name:
                    model_name = 'latest'

                success = self.controller.tagging_engine.load_model(model_name)
                if success:
                    self.root.after(0, lambda: (
                        self.model_status_label.config(text=f"✓ Model loaded ({config_manager.app_config.model_type})", foreground='green'),
                        self.status_var.set(f"Model loaded: {model_name}"),
                        self._refresh_trained_models_async()
                    ))
                else:
                    self.root.after(0, lambda: (
                        self.model_status_label.config(text='⚠ Model failed to load', foreground='orange'),
                        self.status_var.set(f"Model failed to load: {model_name}")
                    ))
            except Exception as e:
                self.root.after(0, lambda: (
                    self.model_status_label.config(text='✗ Load error', foreground='red'),
                    self.status_var.set(f'Model load error: {str(e)[:80]}')
                ))

        threading.Thread(target=_worker, daemon=True).start()
    
    def load_image_tags(self):
        """Load tags for selected image"""
        if self.selected_image_path is None:
            return
        
        self.tags_listbox.delete(0, tk.END)
        tags = self.controller.get_image_tags(self.selected_image_path)
        self.selected_image_tags = tags
        
        for tag in tags:
            confidence = tag.get('confidence', 1.0)
            source = tag.get('source', 'unknown')
            display_text = f"{tag['tag_name']} ({confidence:.2f}) [{source}]"
            self.tags_listbox.insert(tk.END, display_text)
    
    def predict_tags_for_selected(self):
        """Predict tags for currently selected image"""
        if self.selected_image_path is None:
            return
        
        selected_path = self.selected_image_path  # Store in local variable for type checker
        
        self.predictions_listbox.delete(0, tk.END)
        self.predictions_listbox.insert(tk.END, "Predicting...")
        
        def predict_thread():
            try:
                predictions = self.controller.tagging_engine.predict_tags_for_image(selected_path)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.display_predictions(predictions))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Prediction failed: {e}"))
        
        thread = threading.Thread(target=predict_thread, daemon=True)
        thread.start()
    
    def display_predictions(self, predictions):
        """Display predicted tags"""
        self.predictions_listbox.delete(0, tk.END)
        
        if len(predictions) == 0:
            self.predictions_listbox.insert(tk.END, "No predictions (model not loaded?)")
            return
        
        for pred in predictions:
            display_text = f"{pred['tag_name']} ({pred['confidence']:.2f})"
            self.predictions_listbox.insert(tk.END, display_text)
    
    def predict_current_image_tags(self):
        """Toolbar action to predict tags"""
        if self.selected_image_path:
            self.predict_tags_for_selected()
        else:
            messagebox.showinfo("Info", "Please select an image first")
    
    def add_manual_tag(self):
        """Add a manual tag to selected image"""
        if self.selected_image_path is None:
            messagebox.showwarning("Warning", "No image selected")
            return
        
        tag_name = self.new_tag_entry.get().strip()
        if not tag_name:
            messagebox.showwarning("Warning", "Please enter a tag name")
            return
        
        try:
            self.controller.tagging_engine.apply_manual_tag(
                self.selected_image_path, 
                tag_name, 
                is_ground_truth=True
            )
            self.new_tag_entry.delete(0, tk.END)
            self.load_image_tags()
            self.status_var.set(f"Tag '{tag_name}' added")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add tag: {e}")

    def add_folder_tags_to_selected(self):
        """Apply folder-hierarchy tags to the currently selected image."""
        if self.selected_image_path is None:
            messagebox.showwarning("Warning", "No image selected")
            return

        root_folder = config_manager.app_config.tag_root_folder or str(self.root_folder)

        try:
            applied = self.controller.apply_folder_tags(self.selected_image_path, root_folder)
            self.load_image_tags()
            if applied:
                self.status_var.set(f"Folder tags added: {', '.join(applied)}")
            else:
                msg = (f"No folder tags could be derived.\n\n"
                       f"Image: {Path(self.selected_image_path).name}\n"
                       f"Root : {root_folder}\n\n"
                       f"Make sure the image is inside the configured root folder.\n"
                       f"Use File → Set Root Folder to configure it.")
                messagebox.showinfo("No Tags Derived", msg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add folder tags: {e}")
    
    def remove_selected_tag(self):
        """Remove selected tag from image"""
        if self.selected_image_path is None:
            messagebox.showwarning("Warning", "No image selected")
            return
        
        selection = self.tags_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No tag selected")
            return
        
        idx = selection[0]
        tag_data = self.selected_image_tags[idx]
        tag_name = tag_data['tag_name']
        
        try:
            self.controller.tagging_engine.remove_tag(self.selected_image_path, tag_name)
            self.load_image_tags()
            self.status_var.set(f"Tag '{tag_name}' removed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove tag: {e}")
    
    def apply_selected_prediction(self):
        """Apply selected predicted tag"""
        if self.selected_image_path is None:
            messagebox.showwarning("Warning", "No image selected")
            return
        
        selection = self.predictions_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No prediction selected")
            return
        
        # Parse tag name from selection
        pred_text = self.predictions_listbox.get(selection[0])
        tag_name = pred_text.split(' (')[0]
        
        try:
            self.controller.tagging_engine.apply_manual_tag(self.selected_image_path, tag_name)
            self.load_image_tags()
            self.status_var.set(f"Prediction '{tag_name}' applied")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply prediction: {e}")
    
    def open_model_config(self):
        """Open model configuration dialog"""
        config_window = Toplevel(self.root)
        config_window.title("Model Configuration")
        config_window.geometry("500x600")
        
        # Create notebook for different config sections
        notebook = ttk.Notebook(config_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Hyperparameters tab
        hyper_frame = ttk.Frame(notebook)
        notebook.add(hyper_frame, text="Hyperparameters")
        
        # Scrollable frame for hyperparameters
        canvas = Canvas(hyper_frame)
        scrollbar = Scrollbar(hyper_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Hyperparameter fields
        hyper_params = config_manager.hyperparameters
        entries = {}
        
        row = 0
        for field, value in hyper_params.__dict__.items():
            ttk.Label(scrollable_frame, text=f"{field}:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
            entry = ttk.Entry(scrollable_frame, width=20)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, padx=5, pady=5)
            entries[field] = entry
            row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Application config tab
        app_frame = ttk.Frame(notebook)
        notebook.add(app_frame, text="Application")
        
        app_entries = {}
        app_config = config_manager.app_config
        
        row = 0
        for field, value in app_config.__dict__.items():
            ttk.Label(app_frame, text=f"{field}:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
            
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                check = ttk.Checkbutton(app_frame, variable=var)
                check.grid(row=row, column=1, padx=5, pady=5)
                app_entries[field] = var
            else:
                entry = ttk.Entry(app_frame, width=30)
                entry.insert(0, str(value))
                entry.grid(row=row, column=1, padx=5, pady=5)
                app_entries[field] = entry
            row += 1
        
        # Save button
        def save_config():
            try:
                # Save hyperparameters
                for field, entry in entries.items():
                    value = entry.get()
                    # Convert to appropriate type
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except:
                        pass
                    config_manager.update_hyperparameters(**{field: value})
                
                # Save app config
                for field, widget in app_entries.items():
                    if isinstance(widget, tk.BooleanVar):
                        value = widget.get()
                    else:
                        value = widget.get()
                    config_manager.update_app_config(**{field: value})
                
                messagebox.showinfo("Success", "Configuration saved")
                config_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {e}")
        
        ttk.Button(config_window, text="Save Configuration", command=save_config).pack(pady=10)
    
    def _make_training_poll(self, state, progress_dialog, on_complete=None):
        _drain_err_streak = [0]

        def _poll():
            done = False

            # Drain is separated so a drain failure doesn't discard already-queued data.
            try:
                status, progress, metrics, details, terminal = state.drain()
            except Exception as exc:
                print(f"[TrainingPoll] drain error: {exc}")
                _drain_err_streak[0] += 1
                if _drain_err_streak[0] < 10:
                    self.root.after(150, _poll)
                else:
                    self.controller.training_manager.progress_callback = None
                return

            # Commit terminal state BEFORE any UI call that might raise.
            # If a widget error occurs later, the callback is already cleared
            # and the poll will not be rescheduled — no zombie loops.
            if terminal:
                done = True
                self.controller.training_manager.progress_callback = None

            if not progress_dialog._alive():
                return

            _drain_err_streak[0] = 0
            try:
                if status:
                    progress_dialog.update_status(status)
                if progress is not None:
                    progress_dialog.update_progress(progress)
                if metrics:
                    progress_dialog.update_metrics(
                        metrics['train_acc'], metrics['val_acc'], metrics['overfitting_gap'])
                for d in details:
                    progress_dialog.add_detail(d)
                if details:
                    progress_dialog.scroll_to_end()
                if terminal:
                    kind, val = terminal
                    if kind == 'complete':
                        progress_dialog.mark_complete(True)
                        if on_complete:
                            on_complete()
                    elif kind == 'error':
                        progress_dialog.add_detail(f"ERROR: {val}")
                        progress_dialog.scroll_to_end()
                        progress_dialog.mark_complete(False)
                    elif kind == 'cancelled':
                        progress_dialog.mark_cancelled()
            except Exception as exc:
                print(f"[TrainingPoll] UI update error: {exc}")

            if not done:
                self.root.after(150, _poll)
        return _poll

    def train_from_folders(self):
        if not messagebox.askyesno("Confirm",
                "This will train a new model using the folder structure as labels. Continue?"):
            return

        state = _TrainingState()
        progress_dialog = TrainingProgressDialog(
            self.root, "Training from Folders",
            cancel_callback=lambda: self.controller.training_manager.cancel())
        progress_dialog.show()

        def on_complete():
            self.status_var.set("Training completed successfully")
            self.model_status_label.config(
                text=f"✓ Model trained & loaded ({config_manager.app_config.model_type})",
                foreground='green')
            try:
                self._refresh_trained_models_async()
            except Exception:
                pass

        self.controller.training_manager.progress_callback = state.push
        state.push('status', 'Initializing training...')
        state.push('detail', f'Directory: {self.current_path}')

        def _run():
            try:
                self.controller.bootstrap_from_folders()
            except Exception as e:
                state.push('error', str(e))

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(150, self._make_training_poll(state, progress_dialog, on_complete))

    def train_from_tags(self):
        if not messagebox.askyesno("Confirm",
                "This will train a model using existing tags. Continue?"):
            return

        state = _TrainingState()
        progress_dialog = TrainingProgressDialog(
            self.root, "Training from Tags",
            cancel_callback=lambda: self.controller.training_manager.cancel())
        progress_dialog.show()

        def on_complete():
            self.status_var.set("Training completed successfully")
            self.model_status_label.config(
                text=f"✓ Model trained & loaded ({config_manager.app_config.model_type})",
                foreground='green')
            try:
                self._refresh_trained_models_async()
            except Exception:
                pass

        self.controller.training_manager.progress_callback = state.push
        state.push('status', 'Initializing training...')
        state.push('detail', 'Loading tags from database...')

        def _run():
            try:
                self.controller.train_from_corrections()
            except Exception as e:
                state.push('error', str(e))

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(150, self._make_training_poll(state, progress_dialog, on_complete))

    def auto_sort_images(self):
        if not messagebox.askyesno("Confirm",
                "This will predict tags for all images in the current directory. Continue?"):
            return

        state = _TrainingState()
        progress_dialog = TrainingProgressDialog(self.root, "Auto-Sorting Images")
        progress_dialog.show()

        def on_complete():
            self.status_var.set("Auto-sorting completed")

        def _run():
            try:
                results = self.controller.start_initial_sorting(
                    auto_apply=True, progress_callback=state.push)
                if not results:
                    state.push('error', 'No model loaded or no images found. Train a model first.')
            except Exception as e:
                state.push('error', str(e))

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(150, self._make_training_poll(state, progress_dialog, on_complete))
    
    def run_clustering(self):
        """Run unsupervised clustering"""
        n_clusters = simpledialog.askinteger("Clustering", 
            "Number of clusters:", initialvalue=10, minvalue=2, maxvalue=50)
        
        if n_clusters:
            self.status_var.set("Clustering in progress...")
            
            def cluster_thread():
                try:
                    self.controller.perform_unsupervised_clustering(n_clusters)
                    self.root.after(0, lambda: messagebox.showinfo("Complete", 
                        f"Clustering completed with {n_clusters} clusters"))
                    self.root.after(0, lambda: self.status_var.set("Clustering complete"))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Clustering failed: {e}"))
            
            thread = threading.Thread(target=cluster_thread, daemon=True)
            thread.start()
    
    def show_metrics(self):
        """Show performance metrics"""
        metrics_window = Toplevel(self.root)
        metrics_window.title("Performance Metrics")
        metrics_window.geometry("600x500")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(metrics_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = Scrollbar(text_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Get metrics summary
        summary = metrics_collector.get_summary()
        
        metrics_text = "PERFORMANCE METRICS SUMMARY\n"
        metrics_text += "=" * 50 + "\n\n"
        metrics_text += f"Session ID: {summary['session_id']}\n"
        metrics_text += f"Duration: {summary['duration']:.2f} seconds\n\n"
        metrics_text += "INFERENCE METRICS:\n"
        metrics_text += f"  Total Images Processed: {summary['images_processed']}\n"
        metrics_text += f"  Avg Latency per Image: {summary['avg_inference_latency']*1000:.2f} ms\n\n"
        metrics_text += "TRAINING METRICS:\n"
        metrics_text += f"  Total Training Time: {summary['total_training_time']:.2f} seconds\n\n"
        metrics_text += "ACCURACY METRICS:\n"
        metrics_text += f"  Overall Accuracy: {summary['overall_accuracy']:.2f}%\n"
        metrics_text += f"  Precision: {summary['precision']:.4f}\n"
        metrics_text += f"  Recall: {summary['recall']:.4f}\n"
        metrics_text += f"  F1 Score: {summary['f1_score']:.4f}\n"
        
        text_widget.insert(tk.END, metrics_text)
        text_widget.config(state=tk.DISABLED)
        
        ttk.Button(metrics_window, text="Close", command=metrics_window.destroy).pack(pady=10)


def main():
    """Main entry point"""
    if _TKDND_AVAILABLE and _TkDnD is not None:
        root: _Any = _TkDnD.Tk()
    else:
        root = tk.Tk()
    app = FileExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
