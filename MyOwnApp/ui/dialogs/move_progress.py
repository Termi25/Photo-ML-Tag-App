"""MoveProgressDialog — non-blocking progress dialog for file move / copy operations."""

import tkinter as tk
from tkinter import ttk, Toplevel


class MoveProgressDialog:
    """Non-blocking progress dialog for file move / copy operations.

    Created on the UI thread; updated from background threads via ``root.after``.
    Intentionally NOT modal (no ``grab_set``) so the rest of the UI stays responsive.
    """

    def __init__(self, parent, title: str = "Moving Files…", total: int = 0) -> None:
        self.window = Toplevel(parent)
        self.window.title(title)
        self.window.geometry("500x230")
        self.window.resizable(False, False)
        self.window.transient(parent)

        self._total = max(total, 1)

        self.status_label = ttk.Label(self.window, text="Preparing…", font=('Arial', 11, 'bold'))
        self.status_label.pack(pady=(12, 2), padx=16, anchor='w')

        self.file_label = ttk.Label(self.window, text="", font=('Arial', 10), foreground='gray',
                                     wraplength=460, justify='left')
        self.file_label.pack(padx=16, anchor='w')

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.window, variable=self.progress_var,
                                             maximum=100, length=460)
        self.progress_bar.pack(pady=10, padx=16)

        counter_row = ttk.Frame(self.window)
        counter_row.pack(fill=tk.X, padx=16)
        self.counter_label = ttk.Label(counter_row, text=f"0 / {self._total}", font=('Arial', 10))
        self.counter_label.pack(side=tk.LEFT)
        self.error_label = ttk.Label(counter_row, text="", font=('Arial', 10), foreground='red')
        self.error_label.pack(side=tk.RIGHT)

        self.close_btn = ttk.Button(self.window, text="Close",
                                     command=self.window.destroy, state=tk.DISABLED)
        self.close_btn.pack(pady=8)

        self.window.update_idletasks()
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry(f'500x230+{(sw-500)//2}+{(sh-230)//2}')

    def update(self, current_name: str, done: int, errors: int = 0) -> None:
        """Update progress (call via ``root.after`` from background threads)."""
        try:
            self.progress_var.set(min(100.0, done / self._total * 100))
            self.file_label.config(text=current_name)
            self.counter_label.config(text=f"{done} / {self._total}")
            if errors:
                self.error_label.config(text=f"{errors} error(s)")
        except Exception:
            pass

    def mark_done(self, moved: int, skipped: int = 0) -> None:
        try:
            label = f"✓ Done — {moved} moved" + (f", {skipped} skipped" if skipped else "")
            self.status_label.config(text=label)
            self.file_label.config(text="")
            self.progress_var.set(100)
            self.counter_label.config(text=f"{moved} / {self._total}")
            self.close_btn.config(state=tk.NORMAL)
        except Exception:
            pass

    def mark_error(self, message: str) -> None:
        try:
            self.status_label.config(text="✗ Error occurred", foreground='red')
            self.error_label.config(text=message[:70])
            self.close_btn.config(state=tk.NORMAL)
        except Exception:
            pass
