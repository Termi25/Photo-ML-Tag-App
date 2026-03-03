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
import platform
import string
from PIL import Image, ImageTk
import threading
import time

# Ensure the console can display Unicode characters (e.g. ✓ ✗ ⚠) on Windows
# where the default cp1250/cp850 codec would raise UnicodeEncodeError.
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
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD as _TkDnD  # type: ignore[import-untyped]
    _TKDND_AVAILABLE = True
except ImportError:
    _TKDND_AVAILABLE = False

_GALLERY_COLS = 5   # target columns in icon/gallery view
_GALLERY_GAP = 4    # pixels of space between cells (each side gets GAP//2)
_PREVIEW_FILENAME   = '.folder_preview.png'   # static preview stored inside each folder
_PREVIEW_RESOLUTION = 512                      # pixel size of the stored square preview
_PREVIEW_ZOOM       = 1.4                      # zoom-in factor applied to each quadrant tile
_INBOX_FOLDER       = '_INBOX'                 # drop-folder for new unclassified images

# Import application modules
from config import config_manager, HyperParameters
from data_layer import metadata_store, model_storage
from logic_controller import logic_controller
from ml_module import InferenceEngine, TrainingLoop
from metrics import metrics_collector


class TrainingProgressDialog:
    """Dialog to show training progress with real-time updates"""
    
    def __init__(self, parent, title="Training Progress"):
        self.window = Toplevel(parent)
        self.window.title(title)
        self.window.geometry("600x400")
        self.window.resizable(False, False)
        
        # Make it modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Status label
        self.status_label = ttk.Label(self.window, text="Initializing...", 
                                     font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)
        
        # Metrics frame (train vs val accuracy)
        metrics_frame = ttk.Frame(self.window)
        metrics_frame.pack(pady=5)
        
        ttk.Label(metrics_frame, text="Train Acc:", font=('Arial', 12)).grid(row=0, column=0, padx=5)
        self.train_acc_label = ttk.Label(metrics_frame, text="--", font=('Arial', 12, 'bold'), foreground='blue')
        self.train_acc_label.grid(row=0, column=1, padx=5)
        
        ttk.Label(metrics_frame, text="Val Acc:", font=('Arial', 12)).grid(row=0, column=2, padx=5)
        self.val_acc_label = ttk.Label(metrics_frame, text="--", font=('Arial', 12, 'bold'), foreground='green')
        self.val_acc_label.grid(row=0, column=3, padx=5)
        
        ttk.Label(metrics_frame, text="Gap:", font=('Arial', 12)).grid(row=0, column=4, padx=5)
        self.gap_label = ttk.Label(metrics_frame, text="--", font=('Arial', 12, 'bold'))
        self.gap_label.grid(row=0, column=5, padx=5)
        
        # Warning label for overfitting
        self.warning_label = ttk.Label(self.window, text="", font=('Arial', 12), foreground='red')
        self.warning_label.pack(pady=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.window, variable=self.progress_var, 
                                           maximum=100, length=550)
        self.progress_bar.pack(pady=10, padx=25)
        
        # Details frame with scrollbar
        details_frame = ttk.Frame(self.window)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=15, 
                                    font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(details_frame, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons frame
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)
        
        self.close_button = ttk.Button(button_frame, text="Close", 
                                       command=self.close, state=tk.DISABLED)
        self.close_button.pack(side=tk.LEFT, padx=5)
        
        self.is_complete = False
        self.is_cancelled = False
    
    def _alive(self) -> bool:
        """Return True only if the dialog window still exists."""
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def update_status(self, message):
        """Update status label"""
        if self._alive():
            self.status_label.config(text=message)
    
    def update_progress(self, value):
        """Update progress bar (0-100)"""
        if self._alive():
            self.progress_var.set(value)
    
    def add_detail(self, message):
        """Add a detail message to the log"""
        if self._alive():
            self.details_text.insert(tk.END, message + "\n")
            self.details_text.see(tk.END)
    
    def update_metrics(self, train_acc, val_acc, gap):
        """Update the train/val accuracy display"""
        if not self._alive():
            return
        self.train_acc_label.config(text=f"{train_acc:.1f}%")
        self.val_acc_label.config(text=f"{val_acc:.1f}%")
        self.gap_label.config(text=f"{gap:.1f}%")
        
        # Color code the gap
        if gap > 15:
            self.gap_label.config(foreground='red')
            self.warning_label.config(text="⚠ Overfitting detected")
        elif gap > 10:
            self.gap_label.config(foreground='orange')
            self.warning_label.config(text="")
        else:
            self.gap_label.config(foreground='green')
            self.warning_label.config(text="")
    
    def mark_complete(self, success=True):
        """Mark training as complete"""
        self.is_complete = True
        if not self._alive():
            return
        if success:
            self.status_label.config(text="✓ Training Completed Successfully!")
            self.progress_var.set(100)
        else:
            self.status_label.config(text="✗ Training Failed")
        self.close_button.config(state=tk.NORMAL)
        self.window.grab_release()
    
    def close(self):
        """Close the dialog"""
        if self.is_complete or messagebox.askyesno("Cancel", 
                                                    "Training in progress. Cancel?"):
            self.is_cancelled = True
            self.window.destroy()
    
    def show(self):
        """Show the dialog"""
        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.window.winfo_screenheight() // 2) - (400 // 2)
        self.window.geometry(f'600x400+{x}+{y}')


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
            and f.name != _PREVIEW_FILENAME
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
        for img in self.images:
            try:
                preds = self.controller.tagging_engine.predict_tags_for_image(
                    str(img))
                if preds:
                    key = str(img)
                    if key not in self.annotations:
                        self.annotations[key] = [
                            p['tag_name'] for p in preds[:5]]
            except Exception:
                pass
        # Refresh UI after predictions are loaded
        self._populate_image_list()
        if self._selected_idx >= 0:
            self._update_tag_display()
        self._update_bottom_status()

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
            pil_img = Image.open(str(img_path)).convert('RGB')
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


class FileExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("ML-Enhanced Image Sorting Application")
        self.root.geometry("1400x800")
        
        # Current directory - default to ExampleImageFolder
        default_folder = Path(__file__).parent.parent / "ExampleImageFolder"
        if default_folder.exists():
            self.current_path = default_folder
        else:
            self.current_path = Path.home()
        
        # The root folder is the top-level folder the user opened.
        # The shared database lives there; sub-folder navigation does NOT move it.
        self.root_folder = self.current_path
        
        # View mode: 'detailed' or 'icons'
        self.view_mode = 'detailed'

        # Show / hide hidden items (OS hidden attribute + app-hidden flag)
        self.show_hidden_var = tk.BooleanVar(value=False)

        # Store image references to prevent garbage collection
        self.image_references = []
        
        # Current selected image for tagging
        self.selected_image_path = None
        self.selected_image_tags = []
        
        # Initialize logic controller
        self.controller = logic_controller
        
        # Setup UI
        self.setup_ui()
        
        # Initialize controller with current directory
        self.initialize_ml_system()
        
        # Load directory (does NOT move the DB - root was set in initialize_ml_system)
        self.load_directory(self.current_path)
        
    def setup_ui(self):
        """Setup the user interface with ML components"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Change Directory", command=self.navigate_to_path)
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
        
        # ML Control Panel (replaces simple menu)
        self.setup_ml_control_panel()
        
        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Back button
        self.back_btn = ttk.Button(toolbar, text="← Back", command=self.go_back)
        self.back_btn.pack(side=tk.LEFT, padx=2)
        
        # Up button
        self.up_btn = ttk.Button(toolbar, text="↑ Up", command=self.go_up)
        self.up_btn.pack(side=tk.LEFT, padx=2)
        
        # Home button
        self.home_btn = ttk.Button(toolbar, text="🏠 Home", command=self.go_home)
        self.home_btn.pack(side=tk.LEFT, padx=2)
        
        # Refresh button
        self.refresh_btn = ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh)
        self.refresh_btn.pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # View mode buttons
        ttk.Label(toolbar, text="View:").pack(side=tk.LEFT, padx=2)
        self.detailed_btn = ttk.Button(toolbar, text="📋 Detailed", command=self.switch_to_detailed)
        self.detailed_btn.pack(side=tk.LEFT, padx=2)
        self.icons_btn = ttk.Button(toolbar, text="🖼️ Gallery", command=self.switch_to_icons)
        self.icons_btn.pack(side=tk.LEFT, padx=2)
        
        # ML toolbar buttons
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Label(toolbar, text="ML Actions:").pack(side=tk.LEFT, padx=2)
        
        self.predict_btn = ttk.Button(toolbar, text="🤖 Predict Tags", command=self.predict_current_image_tags)
        self.predict_btn.pack(side=tk.LEFT, padx=2)
        
        # Address bar
        ttk.Label(toolbar, text="Path:").pack(side=tk.LEFT, padx=(10, 2))
        self.path_var = tk.StringVar(value=str(self.current_path))
        self.path_entry = ttk.Entry(toolbar, textvariable=self.path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.path_entry.bind('<Return>', lambda e: self.navigate_to_path())
        
        # Go button
        self.go_btn = ttk.Button(toolbar, text="Go", command=self.navigate_to_path)
        self.go_btn.pack(side=tk.LEFT, padx=2)
        
        # Main container with left panel, content area, and right sidebar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for drives
        left_panel = ttk.Frame(main_container, width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        ttk.Label(left_panel, text="Drives", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Drives listbox
        drives_frame = ttk.Frame(left_panel)
        drives_frame.pack(fill=tk.BOTH, expand=True)
        
        self.drives_listbox = tk.Listbox(drives_frame, selectmode=tk.SINGLE)
        drives_scrollbar = ttk.Scrollbar(drives_frame, orient=tk.VERTICAL, command=self.drives_listbox.yview)
        self.drives_listbox.configure(yscrollcommand=drives_scrollbar.set)
        
        self.drives_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        drives_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.drives_listbox.bind('<<ListboxSelect>>', self.on_drive_select)
        
        # Populate drives
        self.populate_drives()
        
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
    
    def setup_ml_control_panel(self):
        """Setup comprehensive ML control panel with standard and advanced sections"""
        # Configure font style (minimum 12pt)
        normal_font = ('Arial', 12)
        bold_font = ('Arial', 12, 'bold')
        advanced_font = ('Arial', 11)
        
        # Main ML panel — standard and advanced sections sit directly in the root
        main_horizontal = ttk.Frame(self.root, relief='groove', borderwidth=1)
        main_horizontal.pack(side=tk.TOP, fill=tk.X, expand=False, padx=5, pady=2)
        
        # ===== LEFT: STANDARD SECTION (independently toggleable) =====
        standard_outer = ttk.Frame(main_horizontal)
        standard_outer.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 2))

        # Standard section header with its own toggle button (no surrounding box)
        standard_header = ttk.Frame(standard_outer)
        standard_header.pack(fill=tk.X)
        self.standard_expanded = tk.BooleanVar(value=True)
        self.standard_toggle_btn = ttk.Button(standard_header, text="▼ Standard",
                                              command=self.toggle_standard_section, width=15)
        self.standard_toggle_btn.pack(side=tk.LEFT, pady=1, padx=1)

        # Standard content frame — plain frame, no LabelFrame box
        self.standard_content = ttk.Frame(standard_outer)
        self.standard_content.pack(fill=tk.BOTH, expand=True)
        
        # Standard content in 2-column grid layout — no scrollbar needed
        standard_grid = ttk.Frame(self.standard_content)
        standard_grid.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
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
        
        # Right column: Actions
        col_right = ttk.Frame(standard_grid)
        col_right.grid(row=0, column=1, sticky='nsew', padx=(1, 0))
        
        ttk.Label(col_right, text="Actions:", font=bold_font).pack(anchor='w')
        
        self.train_folders_btn = ttk.Button(col_right, text="📁 Train Folders", 
                                           command=self.train_from_folders)
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
        
        # Configure grid weights
        standard_grid.grid_columnconfigure(0, weight=1)
        standard_grid.grid_columnconfigure(1, weight=1)
        
        # ===== RIGHT: ADVANCED SECTION (independently toggleable) =====
        advanced_outer = ttk.Frame(main_horizontal)
        advanced_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))

        # Advanced section header with its own toggle button (no surrounding box)
        advanced_header = ttk.Frame(advanced_outer)
        advanced_header.pack(fill=tk.X)
        self.advanced_expanded = tk.BooleanVar(value=True)
        self.advanced_toggle_btn = ttk.Button(advanced_header, text="▼ Advanced",
                                              command=self.toggle_advanced_section, width=15)
        self.advanced_toggle_btn.pack(side=tk.LEFT, pady=1, padx=1)

        # Advanced content frame — plain frame, no LabelFrame box
        self.advanced_content = ttk.Frame(advanced_outer)
        self.advanced_content.pack(fill=tk.BOTH, expand=True)

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
        
        self.auto_load_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Auto-load", variable=self.auto_load_var).pack(anchor='w', pady=1)
        
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
        """Toggle the Standard sub-section inside ML Controls"""
        if self.standard_expanded.get():
            self.standard_content.pack_forget()
            self.standard_toggle_btn.config(text="▶ Standard")
            self.standard_expanded.set(False)
        else:
            self.standard_content.pack(fill=tk.BOTH, expand=True)
            self.standard_toggle_btn.config(text="▼ Standard")
            self.standard_expanded.set(True)

    def toggle_advanced_section(self):
        """Toggle the Advanced sub-section inside ML Controls"""
        if self.advanced_expanded.get():
            self.advanced_content.pack_forget()
            self.advanced_toggle_btn.config(text="▶ Advanced")
            self.advanced_expanded.set(False)
        else:
            self.advanced_content.pack(fill=tk.BOTH, expand=True)
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
            
            messagebox.showinfo("Success", "Hyperparameters applied successfully!\nThese will be used for the next training session.")
            print("✓ Hyperparameters updated")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {e}")
    
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
            self.auto_load_var.set(True)
            
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
    
    def populate_drives(self):
        """Populate the drives list"""
        self.drives_listbox.delete(0, tk.END)
        
        if platform.system() == 'Windows':
            # Get all available drives on Windows
            available_drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    available_drives.append(drive)
            
            for drive in available_drives:
                try:
                    # Try to get drive info
                    total, used, free = self.get_drive_space(drive)
                    label = f"{drive} ({self.format_size(free)} free)"
                    self.drives_listbox.insert(tk.END, label)
                except:
                    self.drives_listbox.insert(tk.END, drive)
        else:
            # For Unix-like systems, show common mount points
            self.drives_listbox.insert(tk.END, "/")
            common_mounts = ["/home", "/media", "/mnt", "/Volumes"]
            for mount in common_mounts:
                if os.path.exists(mount):
                    self.drives_listbox.insert(tk.END, mount)
    
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
    
    def on_drive_select(self, event):
        """Handle drive selection"""
        selection = self.drives_listbox.curselection()
        if selection:
            drive_text = self.drives_listbox.get(selection[0])
            # Extract drive letter/path (before the space if there's additional info)
            drive = drive_text.split()[0]
            self.history.append(self.current_path)
            self.load_directory(Path(drive))
    
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
        
        # Inner frame for icons
        self.icons_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.icons_container, anchor='nw')
        
        # Bind configure event to update scroll region
        self.icons_container.bind('<Configure>', self.on_icons_configure)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        
        # Bind mouse wheel
        self.canvas.bind_all('<MouseWheel>', self.on_mousewheel)
        
        # Store icon buttons for tracking
        self.icon_buttons = []
        
        # Store thumbnail images to prevent garbage collection
        self.tree_thumbnails = {}
        self.thumbnail_cache = {}  # Cache thumbnails for performance

        # Generation counter – incremented on every directory load so that a
        # background thumbnail thread that was started for a previous folder
        # knows to abort rather than updating stale tree items.
        self._thumb_generation = 0

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
            self._gallery_relayout_id = self.root.after(150, self._relayout_gallery)
    
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
    
    def show_icons_view(self):
        """Show the large icons view"""
        self.detailed_frame.pack_forget()
        self.icons_frame.pack(fill=tk.BOTH, expand=True)
        self.view_mode = 'icons'
    
    def switch_to_detailed(self):
        """Switch to detailed view"""
        if self.view_mode != 'detailed':
            self.show_detailed_view()
            self.load_directory(self.current_path)
    
    def switch_to_icons(self):
        """Switch to icons view"""
        if self.view_mode != 'icons':
            self.show_icons_view()
            self.load_directory(self.current_path)
        
    def load_directory(self, path):
        """Load and display directory contents"""
        try:
            path = Path(path)
            if not path.exists():
                messagebox.showerror("Error", f"Path does not exist: {path}")
                return
            
            if not path.is_dir():
                messagebox.showerror("Error", f"Not a directory: {path}")
                return
            
            # Update current path
            self.current_path = path
            self.path_var.set(str(path))
            
            # Do NOT move the database when navigating sub-folders.
            # The DB stays at the root folder set via set_root_folder_dialog().
            
            # Update controller's working directory (keeps model loaded)
            if hasattr(self, 'controller') and self.controller:
                self.controller.change_directory(str(path))
            
            # Get directory contents
            items = []
            try:
                items = list(path.iterdir())
            except PermissionError:
                messagebox.showerror("Error", f"Permission denied: {path}")
                return
            
            # Sort: directories first, then files
            folders = sorted([item for item in items if item.is_dir()], key=lambda x: x.name.lower())
            files = sorted([item for item in items if item.is_file()], key=lambda x: x.name.lower())

            # Filter hidden items unless the user toggled "Show Hidden Items"
            if not self.show_hidden_var.get():
                folders = [f for f in folders if not self._is_os_hidden(f)]
                files   = [f for f in files   if not self._is_os_hidden(f)]

                # Also filter images that are marked hidden inside the app DB
                app_hidden = self._get_app_hidden_paths(str(path))
                if app_hidden:
                    files = [f for f in files if str(f.resolve()) not in app_hidden]

            # Reset scroll positions to top before populating the new directory
            self.canvas.yview_moveto(0)
            self.tree.yview_moveto(0)

            if self.view_mode == 'detailed':
                self.load_detailed_view(folders, files)
            else:
                self.load_icons_view(folders, files)
            
            # Update status
            self.status_var.set(f"{len(folders)} folders, {len(files)} files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load directory: {e}")
    
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
        if cache_key in self.thumbnail_cache:
            return self.thumbnail_cache[cache_key]
        
        try:
            ext = path.suffix.lower()
            
            # For image files, create actual thumbnail
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico', '.jfif']:
                img = Image.open(path)
                
                # Convert RGBA to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Resize to thumbnail size
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Cache the thumbnail
                self.thumbnail_cache[cache_key] = photo
                return photo
            else:
                # For non-image files, create a colored square with file type indicator
                img = Image.new('RGB', (size, size), (240, 240, 240))
                photo = ImageTk.PhotoImage(img)
                self.thumbnail_cache[cache_key] = photo
                return photo
                
        except Exception as e:
            # If thumbnail creation fails, return a default icon
            img = Image.new('RGB', (size, size), (200, 200, 200))
            photo = ImageTk.PhotoImage(img)
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
                        self.thumbnail_cache[ck] = ph
                        self.tree.item(iid, image=ph)
                        self.tree_thumbnails[str(p)] = ph
                    except Exception:
                        pass
                self.root.after(0, _apply)

        threading.Thread(target=_load_thumbs, daemon=True).start()
    
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
        stale_keys = [k for k in self.thumbnail_cache
                      if isinstance(k, tuple) and len(k) == 3 and k[0] == 'folder_collage']
        for k in stale_keys:
            del self.thumbnail_cache[k]

        msg = f"Deleted {deleted} folder preview file(s)."
        if errors:
            msg += f" ({errors} could not be deleted.)"
        messagebox.showinfo("Reset Folder Previews", msg)
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
                if self._is_os_hidden(item):
                    continue
                if item.name == _PREVIEW_FILENAME:
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
                 if f.is_dir() and not self._is_os_hidden(f) and f.name != _PREVIEW_FILENAME],
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

        # --- Try to load an existing, up-to-date static preview ---
        if preview_file.exists():
            try:
                folder_mtime  = folder_path.stat().st_mtime
                preview_mtime = preview_file.stat().st_mtime
                if folder_mtime <= preview_mtime:
                    img = Image.open(preview_file).convert('RGB')
                    img = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    return img
            except Exception:
                pass  # fall through to rebuild

        # --- Build fresh collage at the fixed storage resolution ---
        res  = _PREVIEW_RESOLUTION
        half = res // 2
        collage = Image.new('RGB', (res, res), (210, 210, 210))
        preview_images = self._collect_folder_preview_images(folder_path, 4)
        positions = [(0, 0), (half, 0), (0, half), (half, half)]
        for i, img_path in enumerate(preview_images):
            try:
                img = Image.open(img_path).convert('RGB')
                # zoom-crop with extra zoom-in factor (no gaps/margins between tiles)
                tile = self._zoom_fill(img, half, zoom=_PREVIEW_ZOOM)
                collage.paste(tile, positions[i])
            except Exception:
                pass

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

        return collage.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)

    def _gallery_thumb_size(self):
        """Compute thumbnail size so _GALLERY_COLS fit the canvas width with minimal waste."""
        w = self.canvas.winfo_width()
        if w < 20:
            w = 600  # fallback before canvas is realised
        gap_total = _GALLERY_GAP * (_GALLERY_COLS + 1)
        return max(40, (w - gap_total) // _GALLERY_COLS)

    def _relayout_gallery(self):
        """Re-render the gallery when canvas width changes enough to matter."""
        self._gallery_relayout_id = None
        if not hasattr(self, '_gallery_items') or self._gallery_items is None:
            return
        new_size = self._gallery_thumb_size()
        if abs(new_size - getattr(self, '_gallery_last_thumb_size', 0)) > 2:
            folders, files = self._gallery_items
            self.load_icons_view(folders, files)

    def load_icons_view(self, folders, files):
        # Store items so the canvas-resize handler can re-layout.
        self._gallery_items = (folders, files)

        # Bump generation so any running thumbnail thread aborts.
        self._thumb_generation += 1
        current_gen = self._thumb_generation

        # Compute thumbnail size to fill the canvas width with _GALLERY_COLS columns.
        thumb_size = self._gallery_thumb_size()
        self._gallery_last_thumb_size = thumb_size

        # Clear existing icon buttons
        for widget in self.icons_container.winfo_children():
            widget.destroy()
        self.icon_buttons = []
        self.image_references = []  # Clear image references

        all_items = folders + files

        # --- Phase 1: insert all items synchronously with placeholders ---
        # pending_thumbs collects (label_widget, path) for images that still
        # need a real thumbnail.
        pending_thumbs: list[tuple] = []
        # pending_folders collects (label_widget, path) for folder collage previews.
        pending_folders: list[tuple] = []

        col = 0
        row = 0
        max_cols = _GALLERY_COLS

        for item in all_items:
            icon_frame, placeholder_label = self._create_icon_widget_fast(item, thumb_size)
            icon_frame.grid(row=row, column=col,
                            padx=_GALLERY_GAP // 2, pady=_GALLERY_GAP // 2,
                            sticky='nsew')
            self.icon_buttons.append((icon_frame, item))

            if placeholder_label is not None:
                if item.is_dir():
                    pending_folders.append((placeholder_label, item))
                else:
                    pending_thumbs.append((placeholder_label, item))

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Give each column equal weight so cells fill the full canvas width.
        for c in range(_GALLERY_COLS):
            self.icons_container.grid_columnconfigure(c, weight=1, uniform='gal_col')

        if not pending_thumbs and not pending_folders:
            return

        # --- Phase 2: load and apply thumbnails on a background thread ---
        def _load_thumbs():
            for lbl, path in pending_thumbs:
                if self._thumb_generation != current_gen:
                    return  # navigated away

                cache_key = (str(path), thumb_size)
                cached = self.thumbnail_cache.get(cache_key)

                if cached is not None:
                    def _apply_cached(widget=lbl, ph=cached):
                        if self._thumb_generation != current_gen:
                            return
                        try:
                            widget.config(image=ph, text='')
                            widget.image = ph  # type: ignore
                            self.image_references.append(ph)
                        except Exception:
                            pass
                    self.root.after(0, _apply_cached)
                    continue

                # Build PIL image on background thread
                try:
                    img = Image.open(path)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        bg.paste(
                            img,
                            mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None
                        )
                        img = bg
                    img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                except Exception:
                    continue  # keep the grey placeholder

                # Create PhotoImage and update widget on main thread
                def _apply(widget=lbl, pil=img, ck=cache_key):
                    if self._thumb_generation != current_gen:
                        return
                    try:
                        ph = ImageTk.PhotoImage(pil)
                        self.thumbnail_cache[ck] = ph
                        widget.config(image=ph, text='')
                        widget.image = ph  # type: ignore
                        self.image_references.append(ph)
                    except Exception:
                        pass
                self.root.after(0, _apply)

            for lbl, path in pending_folders:
                if self._thumb_generation != current_gen:
                    return

                cache_key = ('folder_collage', str(path), thumb_size)
                cached = self.thumbnail_cache.get(cache_key)

                if cached is not None:
                    def _apply_folder_cached(widget=lbl, ph=cached):
                        if self._thumb_generation != current_gen:
                            return
                        try:
                            widget.config(image=ph, text='')
                            widget.image = ph  # type: ignore
                            self.image_references.append(ph)
                        except Exception:
                            pass
                    self.root.after(0, _apply_folder_cached)
                    continue

                # Build PIL collage on background thread
                try:
                    collage = self._build_folder_collage(path, thumb_size)
                except Exception:
                    continue  # keep the grey placeholder

                # Create PhotoImage and update widget on main thread
                def _apply_folder(widget=lbl, pil=collage, ck=cache_key):
                    if self._thumb_generation != current_gen:
                        return
                    try:
                        ph = ImageTk.PhotoImage(pil)
                        self.thumbnail_cache[ck] = ph
                        widget.config(image=ph, text='')
                        widget.image = ph  # type: ignore
                        self.image_references.append(ph)
                    except Exception:
                        pass
                self.root.after(0, _apply_folder)

        threading.Thread(target=_load_thumbs, daemon=True).start()

    def _create_icon_widget_fast(self, path, thumb_size=120):
        """Create an icon frame with an *emoji* placeholder immediately.

        Returns ``(frame, label_or_None)``.
        If the item is an image file, also returns the label so the caller can
        later swap in a real thumbnail.  Returns None as the label for folders
        and non-image files.
        """
        frame = ttk.Frame(self.icons_container)

        is_image = (
            path.is_file()
            and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif',
                                         '.bmp', '.webp', '.tiff', '.ico', '.jfif'}
        )

        if is_image:
            # Check cache first — avoids the background wait for already-seen images
            cache_key = (str(path), thumb_size)
            cached = self.thumbnail_cache.get(cache_key)
            if cached is not None:
                icon_label = tk.Label(frame, image=cached, bg='white')
                icon_label.image = cached  # type: ignore
                icon_label.pack()
                self.image_references.append(cached)
                thumb_label = None  # no background work needed
            else:
                # Placeholder: grey square so the grid layout is correct immediately
                placeholder = ImageTk.PhotoImage(
                    Image.new('RGB', (thumb_size, thumb_size), (220, 220, 220))
                )
                self.image_references.append(placeholder)
                icon_label = tk.Label(frame, image=placeholder, bg='white')
                icon_label.image = placeholder  # type: ignore
                icon_label.pack()
                thumb_label = icon_label  # background thread will update this
        elif path.is_dir():
            # Check collage cache first
            cache_key = ('folder_collage', str(path), thumb_size)
            cached = self.thumbnail_cache.get(cache_key)
            if cached is not None:
                icon_label = tk.Label(frame, image=cached, bg='white')
                icon_label.image = cached  # type: ignore
                icon_label.pack()
                self.image_references.append(cached)
                thumb_label = None
            else:
                placeholder = ImageTk.PhotoImage(
                    Image.new('RGB', (thumb_size, thumb_size), (210, 210, 210))
                )
                self.image_references.append(placeholder)
                icon_label = tk.Label(frame, image=placeholder, bg='white')
                icon_label.image = placeholder  # type: ignore
                icon_label.pack()
                thumb_label = icon_label  # background thread will replace with collage
        else:
            file_type_display, _ = self.get_file_type_from_extension(path)
            icon_text = file_type_display.split()[0] if file_type_display else '📄'
            icon_label = tk.Label(frame, text=icon_text, font=('Arial', 48), bg='white')
            icon_label.pack()
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
        for p in paths:
            try:
                if p.is_dir():
                    shutil.rmtree(str(p))
                else:
                    p.unlink()
            except Exception as e:
                messagebox.showerror('Delete Error', str(e))
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
            self.load_directory(previous_path)
    
    def go_up(self):
        """Go to parent directory"""
        parent = self.current_path.parent
        if parent != self.current_path:  # Not at root
            self.history.append(self.current_path)
            self.load_directory(parent)
    
    def go_home(self):
        """Go to home directory"""
        self.history.append(self.current_path)
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
        """Setup the compact tag management sidebar."""
        tag_sidebar = ttk.Frame(parent, width=260)
        tag_sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        tag_sidebar.pack_propagate(False)
        self.tag_sidebar = tag_sidebar  # keep reference so on_mousewheel can detect it

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

    def initialize_ml_system(self):
        """Initialize the ML system in background"""
        def init_thread():
            try:
                # Set the root folder first — this places the DB in the root
                self.controller.set_root_folder(str(self.root_folder))
                self._ensure_inbox_exists()  # create _INBOX if needed
                self.controller.initialize(str(self.current_path))
                # Update status based on whether model was loaded
                if self.controller.tagging_engine.model_loaded:
                    self.status_var.set("✓ ML system ready - Model loaded")
                    # Update ML control panel status
                    self.model_status_label.config(text=f"✓ Model loaded ({config_manager.app_config.model_type})", 
                                                   foreground='green')
                else:
                    self.status_var.set("⚠ ML system ready - No model loaded (train a model first)")
                    self.model_status_label.config(text="⚠ No model loaded", foreground='orange')
            except Exception as e:
                self.status_var.set(f"ML initialization error: {str(e)[:50]}")
                self.model_status_label.config(text="✗ Initialization error", foreground='red')
        
        thread = threading.Thread(target=init_thread, daemon=True)
        thread.start()
    
    def on_image_selected(self, image_path: str):
        """Handle image selection for tagging"""
        self.selected_image_path = image_path
        path_obj = Path(image_path)
        self.selected_image_label.config(text=f"Selected: .../{path_obj.name}")
        
        # Load current tags
        self.load_image_tags()
        
        # Auto-predict if configured
        if config_manager.app_config.supervised_mode:
            self.predict_tags_for_selected()
    
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
    
    def train_from_folders(self):
        """Train model from folder structure with progress dialog"""
        if not messagebox.askyesno("Confirm", 
            "This will train a new model using the folder structure as labels. Continue?"):
            return
        
        # Create progress dialog
        progress_dialog = TrainingProgressDialog(self.root, "Training from Folders")
        progress_dialog.show()
        
        # Setup progress callback
        def progress_callback(msg_type, message):
            if msg_type == 'status':
                self.root.after(0, lambda: progress_dialog.update_status(message))
            elif msg_type == 'detail':
                self.root.after(0, lambda: progress_dialog.add_detail(message))
            elif msg_type == 'progress':
                self.root.after(0, lambda: progress_dialog.update_progress(message))
            elif msg_type == 'metrics':
                self.root.after(0, lambda m=message: progress_dialog.update_metrics(
                    m['train_acc'], m['val_acc'], m['overfitting_gap']))
            elif msg_type == 'complete':
                self.root.after(0, lambda: progress_dialog.mark_complete(True))
                self.root.after(0, lambda: self.status_var.set("Training completed successfully"))
                self.root.after(0, lambda: self.model_status_label.config(
                    text=f"✓ Model trained & loaded ({config_manager.app_config.model_type})", foreground='green'))
            elif msg_type == 'error':
                self.root.after(0, lambda: progress_dialog.add_detail(f"ERROR: {message}"))
                self.root.after(0, lambda: progress_dialog.mark_complete(False))
        
        def train_thread():
            try:
                # Set the progress callback
                self.controller.training_manager.progress_callback = progress_callback
                progress_callback('status', 'Initializing training...')
                progress_callback('detail', f'Directory: {self.current_path}')
                
                self.controller.bootstrap_from_folders()
                
                # Wait for training to complete
                while self.controller.training_manager.is_training:
                    time.sleep(0.5)
                
            except Exception as e:
                self.root.after(0, lambda: progress_callback('error', str(e)))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
        
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()
    
    def train_from_tags(self):
        """Train model from database tags with progress dialog"""
        if not messagebox.askyesno("Confirm", 
            "This will train a model using existing tags. Continue?"):
            return
        
        # Create progress dialog
        progress_dialog = TrainingProgressDialog(self.root, "Training from Tags")
        progress_dialog.show()
        
        # Setup progress callback
        def progress_callback(msg_type, message):
            if msg_type == 'status':
                self.root.after(0, lambda: progress_dialog.update_status(message))
            elif msg_type == 'detail':
                self.root.after(0, lambda: progress_dialog.add_detail(message))
            elif msg_type == 'progress':
                self.root.after(0, lambda: progress_dialog.update_progress(message))
            elif msg_type == 'metrics':
                self.root.after(0, lambda m=message: progress_dialog.update_metrics(
                    m['train_acc'], m['val_acc'], m['overfitting_gap']))
            elif msg_type == 'complete':
                self.root.after(0, lambda: progress_dialog.mark_complete(True))
                self.root.after(0, lambda: self.status_var.set("Training completed successfully"))
            elif msg_type == 'error':
                self.root.after(0, lambda: progress_dialog.add_detail(f"ERROR: {message}"))
                self.root.after(0, lambda: progress_dialog.mark_complete(False))
        
        def train_thread():
            try:
                # Set the progress callback
                self.controller.training_manager.progress_callback = progress_callback
                progress_callback('status', 'Initializing training...')
                progress_callback('detail', 'Loading tags from database...')
                
                self.controller.train_from_corrections()
                
                # Wait for training to complete
                while self.controller.training_manager.is_training:
                    time.sleep(0.5)
                
                # Update model status after successful completion
                self.root.after(0, lambda: self.model_status_label.config(
                    text=f"✓ Model trained & loaded ({config_manager.app_config.model_type})", foreground='green'))
                
            except Exception as e:
                self.root.after(0, lambda: progress_callback('error', str(e)))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
        
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()
    
    def auto_sort_images(self):
        """Auto-sort all images in current directory with progress dialog"""
        if not messagebox.askyesno("Confirm", 
            "This will predict tags for all images in the current directory. Continue?"):
            return
        
        # Create progress dialog
        progress_dialog = TrainingProgressDialog(self.root, "Auto-Sorting Images")
        progress_dialog.show()
        
        # Setup progress callback
        def progress_callback(msg_type, message):
            if msg_type == 'status':
                self.root.after(0, lambda: progress_dialog.update_status(message))
            elif msg_type == 'detail':
                self.root.after(0, lambda: progress_dialog.add_detail(message))
            elif msg_type == 'progress':
                self.root.after(0, lambda: progress_dialog.update_progress(message))
            elif msg_type == 'complete':
                self.root.after(0, lambda: progress_dialog.mark_complete(True))
                self.root.after(0, lambda: self.status_var.set("Auto-sorting completed"))
            elif msg_type == 'error':
                self.root.after(0, lambda: progress_dialog.add_detail(f"ERROR: {message}"))
                self.root.after(0, lambda: progress_dialog.mark_complete(False))
        
        def sort_thread():
            try:
                progress_callback('status', 'Starting auto-sorting...')
                results = self.controller.start_initial_sorting(auto_apply=True, progress_callback=progress_callback)
                
                if not results:
                    progress_callback('error', 'No model loaded or no images to process')
                    self.root.after(0, lambda: messagebox.showerror("Error", 
                        "Failed to start auto-sorting. Make sure a model is trained and loaded."))
            except Exception as e:
                progress_callback('error', str(e))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Auto-sorting failed: {e}"))
        
        thread = threading.Thread(target=sort_thread, daemon=True)
        thread.start()
    
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
