import tkinter as tk
from tkinter import ttk, Toplevel


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
        sb = ttk.Scrollbar(details_frame, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=sb.set)
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._detail_line_count = 0

        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=8)
        self._cancel_btn = ttk.Button(btn_frame, text="Cancel",
                                      command=self._on_cancel,
                                      state=tk.NORMAL if cancel_callback else tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=6)
        self._close_btn = ttk.Button(btn_frame, text="Close",
                                     command=self.window.destroy, state=tk.DISABLED)
        self._close_btn.pack(side=tk.LEFT, padx=6)

        self.is_complete  = False
        self.is_cancelled = False

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
            text="✓ Training Completed Successfully!" if success else "✗ Training Failed"
        )
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
