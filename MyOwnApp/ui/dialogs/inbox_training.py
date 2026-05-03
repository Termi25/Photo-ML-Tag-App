"""InboxTrainingDialog — tag inbox images and queue a fine-tune pass."""

import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Toplevel
from pathlib import Path
from PIL import Image, ImageTk

# Folder-preview thumbnail name (kept in sync with file_explorer constants)
_PREVIEW_FILENAME = '.folder_preview.png'


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

    def __init__(self, parent, inbox_path: Path, controller, known_tags: list) -> None:
        self.parent     = parent
        self.inbox_path = inbox_path
        self.controller = controller
        self.known_tags = sorted(known_tags)
        self.annotations: dict = {}

        self.images: list[Path] = sorted(
            f for f in inbox_path.iterdir()
            if f.is_file() and f.suffix.lower() in self._IMG_EXTS
            and f.name != _PREVIEW_FILENAME
        )

        self._thumb_refs:   list = []
        self._preview_ref        = None
        self._selected_idx: int  = -1

        self.window = Toplevel(parent)
        self.window.title(f"Inbox: Tag & Fine-Tune  ({len(self.images)} image(s))")
        self.window.geometry('960x620')
        self.window.resizable(True, True)
        self.window.transient(parent)
        self.window.grab_set()

        self._build_ui()
        self._populate_image_list()
        if self.images:
            self._select_image(0)

        self.window.update_idletasks()
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry(f'960x620+{(sw-960)//2}+{(sh-620)//2}')

        if controller.tagging_engine.model_loaded and self.images:
            self.window.after(100, self._prepopulate_predictions)

    # ------------------------------------------------------------------
    # Pre-population
    # ------------------------------------------------------------------

    def _prepopulate_predictions(self) -> None:
        if not self.controller.tagging_engine.model_loaded:
            return
        for img in self.images:
            try:
                preds = self.controller.tagging_engine.predict_tags_for_image(str(img))
                if preds:
                    key = str(img)
                    if key not in self.annotations:
                        self.annotations[key] = [p['tag_name'] for p in preds[:5]]
            except Exception:
                pass
        self._populate_image_list()
        if self._selected_idx >= 0:
            self._update_tag_display()
        self._update_bottom_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        ttk.Label(
            self.window,
            text=("Tag each inbox image, then press 'Fine-Tune Model'. "
                  "Only tags the model already knows are used — "
                  "unknown tags are shown in orange as a reminder."),
            font=('Arial', 9), foreground='gray', wraplength=940,
        ).pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Separator(self.window, orient=tk.HORIZONTAL).pack(fill=tk.X)

        pane = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: image list
        left = ttk.Frame(pane, width=230)
        pane.add(left, weight=1)
        ttk.Label(left, text='Images', font=('Arial', 10, 'bold')).pack(anchor='w', padx=4, pady=(2, 0))
        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self._img_listbox = tk.Listbox(
            list_frame, font=('Consolas', 9), selectmode=tk.SINGLE,
            activestyle='none', selectbackground='#4a90d9', selectforeground='white',
        )
        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._img_listbox.yview)
        self._img_listbox.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._img_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._img_listbox.bind('<<ListboxSelect>>', self._on_list_select)

        # Right: preview + tag editor
        right = ttk.Frame(pane)
        pane.add(right, weight=3)

        self._canvas = Canvas(right, width=self._PREVIEW_SZ, height=self._PREVIEW_SZ,
                               bg='#2b2b2b', highlightthickness=0)
        self._canvas.pack(pady=(4, 2))

        self._img_name_var = tk.StringVar(value='')
        ttk.Label(right, textvariable=self._img_name_var, font=('Arial', 9, 'bold')).pack()

        ttk.Label(right, text='Tags:', font=('Arial', 9, 'bold')).pack(anchor='w', padx=8, pady=(4, 0))
        self._chips_frame = ttk.Frame(right)
        self._chips_frame.pack(fill=tk.X, padx=8, pady=2)

        add_row = ttk.Frame(right)
        add_row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(add_row, text='Add tag:', font=('Arial', 9)).pack(side=tk.LEFT)
        self._tag_var   = tk.StringVar()
        self._tag_entry = ttk.Entry(add_row, textvariable=self._tag_var, width=22)
        self._tag_entry.pack(side=tk.LEFT, padx=4)
        self._tag_entry.bind('<Return>', lambda e: self._add_tag_from_entry())
        ttk.Button(add_row, text='+Add', command=self._add_tag_from_entry).pack(side=tk.LEFT)

        ttk.Label(right, text='Known tags (double-click to add):', font=('Arial', 9)).pack(
            anchor='w', padx=8, pady=(6, 0))
        kt_frame = ttk.Frame(right)
        kt_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        self._kt_var      = tk.StringVar(value=self.known_tags)  # type: ignore[arg-type]
        self._kt_listbox  = tk.Listbox(kt_frame, listvariable=self._kt_var, font=('Consolas', 9),
                                        height=6, selectmode=tk.SINGLE, exportselection=False)
        kt_vsb = ttk.Scrollbar(kt_frame, orient=tk.VERTICAL, command=self._kt_listbox.yview)
        self._kt_listbox.configure(yscrollcommand=kt_vsb.set)
        kt_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._kt_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._kt_listbox.bind('<Double-Button-1>', self._add_tag_from_browser)

        filter_row = ttk.Frame(right)
        filter_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(filter_row, text='Filter:', font=('Arial', 9)).pack(side=tk.LEFT)
        self._filter_var = tk.StringVar()
        self._filter_var.trace_add('write', self._filter_known_tags)
        ttk.Entry(filter_row, textvariable=self._filter_var, width=20).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self.window, orient=tk.HORIZONTAL).pack(fill=tk.X)
        bot = ttk.Frame(self.window)
        bot.pack(fill=tk.X, padx=8, pady=6)
        self._status_var = tk.StringVar(value='')
        ttk.Label(bot, textvariable=self._status_var, font=('Arial', 9), foreground='gray').pack(side=tk.LEFT)
        ttk.Label(bot, text='  Epochs:', font=('Arial', 9)).pack(side=tk.LEFT)
        self._epochs_var = tk.IntVar(value=5)
        ttk.Spinbox(bot, from_=1, to=20, width=4, textvariable=self._epochs_var).pack(side=tk.LEFT, padx=4)
        self._ft_btn = ttk.Button(bot, text='Fine-Tune Model (0 labeled)', command=self._start_fine_tune)
        self._ft_btn.pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(bot, text='Close', command=self.window.destroy).pack(side=tk.RIGHT)
        self._update_bottom_status()

    # ------------------------------------------------------------------
    # Image list helpers
    # ------------------------------------------------------------------

    def _populate_image_list(self) -> None:
        self._img_listbox.delete(0, tk.END)
        for img in self.images:
            n    = len(self.annotations.get(str(img), []))
            mark = '✓' if n > 0 else '○'
            self._img_listbox.insert(tk.END, f' {mark} {img.name}  [{n}]')

    def _refresh_list_item(self, idx: int) -> None:
        img  = self.images[idx]
        n    = len(self.annotations.get(str(img), []))
        mark = '✓' if n > 0 else '○'
        self._img_listbox.delete(idx)
        self._img_listbox.insert(idx, f' {mark} {img.name}  [{n}]')
        self._img_listbox.selection_set(idx)

    def _on_list_select(self, _event) -> None:
        sel = self._img_listbox.curselection()
        if sel:
            self._select_image(int(sel[0]))

    def _select_image(self, idx: int) -> None:
        self._selected_idx = idx
        img_path = self.images[idx]
        self._img_name_var.set(img_path.name)
        try:
            with Image.open(str(img_path)) as raw:
                pil = raw.convert('RGB')
            pil.thumbnail((self._PREVIEW_SZ, self._PREVIEW_SZ), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            self._preview_ref = photo
            self._canvas.delete('all')
            cx = cy = self._PREVIEW_SZ // 2
            self._canvas.create_image(cx, cy, image=photo, anchor='center')
        except Exception:
            self._canvas.delete('all')
            self._canvas.create_text(self._PREVIEW_SZ // 2, self._PREVIEW_SZ // 2,
                                     text='(preview unavailable)', fill='gray')
        self._update_tag_display()

    # ------------------------------------------------------------------
    # Tag management
    # ------------------------------------------------------------------

    def _update_tag_display(self) -> None:
        for w in self._chips_frame.winfo_children():
            w.destroy()
        if self._selected_idx < 0:
            return
        img_key = str(self.images[self._selected_idx])
        tags    = list(self.annotations.get(img_key, []))
        for tag in tags:
            colour = 'orange' if tag not in self.known_tags else '#4a90d9'
            chip   = tk.Frame(self._chips_frame, bg=colour, bd=0, padx=4, pady=2)
            chip.pack(side=tk.LEFT, padx=2, pady=2)
            tk.Label(chip, text=tag, bg=colour, fg='white', font=('Arial', 9)).pack(side=tk.LEFT)
            tk.Button(chip, text='×', bg=colour, fg='white', font=('Arial', 8, 'bold'),
                      bd=0, cursor='hand2',
                      command=lambda t=tag: self._remove_tag(t)).pack(side=tk.LEFT)
        if not tags:
            ttk.Label(self._chips_frame, text='(no tags)', foreground='gray',
                      font=('Arial', 9)).pack(side=tk.LEFT)

    def _add_tag_from_entry(self) -> None:
        tag = self._tag_var.get().strip()
        if tag:
            self._add_tag(tag)
            self._tag_var.set('')

    def _add_tag_from_browser(self, _event=None) -> None:
        sel = self._kt_listbox.curselection()
        if sel:
            self._add_tag(self._kt_listbox.get(int(sel[0])))

    def _add_tag(self, tag: str) -> None:
        if self._selected_idx < 0 or not tag:
            return
        img_key = str(self.images[self._selected_idx])
        tags    = self.annotations.setdefault(img_key, [])
        if tag not in tags:
            tags.append(tag)
        self._update_tag_display()
        self._refresh_list_item(self._selected_idx)
        self._update_bottom_status()

    def _remove_tag(self, tag: str) -> None:
        if self._selected_idx < 0:
            return
        img_key = str(self.images[self._selected_idx])
        tags    = self.annotations.get(img_key, [])
        if tag in tags:
            tags.remove(tag)
        if not tags:
            self.annotations.pop(img_key, None)
        self._update_tag_display()
        self._refresh_list_item(self._selected_idx)
        self._update_bottom_status()

    def _filter_known_tags(self, *_) -> None:
        q        = self._filter_var.get().strip().lower()
        filtered = [t for t in self.known_tags if q in t.lower()] if q else self.known_tags
        self._kt_var.set(filtered)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Bottom bar
    # ------------------------------------------------------------------

    def _update_bottom_status(self) -> None:
        labeled = sum(1 for v in self.annotations.values() if v)
        self._status_var.set(f'{labeled} of {len(self.images)} image(s) labeled')
        self._ft_btn.config(
            text=f'Fine-Tune Model ({labeled} labeled)',
            state=tk.NORMAL if labeled > 0 else tk.DISABLED,
        )

    def _start_fine_tune(self) -> None:
        labeled = {k: v for k, v in self.annotations.items() if v}
        if not labeled:
            messagebox.showwarning('No labels',
                'Please tag at least one image before fine-tuning.', parent=self.window)
            return
        epochs = max(1, self._epochs_var.get())
        if not messagebox.askyesno(
            'Fine-Tune Model',
            f'Fine-tune the model on {len(labeled)} labeled image(s) for {epochs} epoch(s)?\n\n'
            'This runs in the background. The current model will be updated when finished.',
            parent=self.window,
        ):
            return
        self.controller.fine_tune_from_inbox_labels(labeled, epochs=epochs)
        self._status_var.set(f'Fine-tuning queued ({len(labeled)} image(s), {epochs} epoch(s))…')
        self._ft_btn.config(state=tk.DISABLED)
        messagebox.showinfo('Queued',
            'Fine-tuning has been queued and will run in the background.\n'
            'Check the status bar for progress.', parent=self.window)
