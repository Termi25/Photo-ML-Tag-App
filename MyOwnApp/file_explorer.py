"""
ML-Enhanced Image Sorting Application
4-tier architecture with supervised/unsupervised learning capabilities
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Toplevel, Scrollbar, simpledialog
from pathlib import Path
import platform
import string
from PIL import Image, ImageTk
import threading
import time

# Import application modules
from config import config_manager
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
                                     font=('Arial', 11, 'bold'))
        self.status_label.pack(pady=10)
        
        # Metrics frame (train vs val accuracy)
        metrics_frame = ttk.Frame(self.window)
        metrics_frame.pack(pady=5)
        
        ttk.Label(metrics_frame, text="Train Acc:", font=('Arial', 9)).grid(row=0, column=0, padx=5)
        self.train_acc_label = ttk.Label(metrics_frame, text="--", font=('Arial', 9, 'bold'), foreground='blue')
        self.train_acc_label.grid(row=0, column=1, padx=5)
        
        ttk.Label(metrics_frame, text="Val Acc:", font=('Arial', 9)).grid(row=0, column=2, padx=5)
        self.val_acc_label = ttk.Label(metrics_frame, text="--", font=('Arial', 9, 'bold'), foreground='green')
        self.val_acc_label.grid(row=0, column=3, padx=5)
        
        ttk.Label(metrics_frame, text="Gap:", font=('Arial', 9)).grid(row=0, column=4, padx=5)
        self.gap_label = ttk.Label(metrics_frame, text="--", font=('Arial', 9, 'bold'))
        self.gap_label.grid(row=0, column=5, padx=5)
        
        # Warning label for overfitting
        self.warning_label = ttk.Label(self.window, text="", font=('Arial', 9), foreground='red')
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
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.window.update()
    
    def update_progress(self, value):
        """Update progress bar (0-100)"""
        self.progress_var.set(value)
        self.window.update()
    
    def add_detail(self, message):
        """Add a detail message to the log"""
        self.details_text.insert(tk.END, message + "\n")
        self.details_text.see(tk.END)
        self.window.update()
    
    def update_metrics(self, train_acc, val_acc, gap):
        """Update the train/val accuracy display"""
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
        
        self.window.update()
    
    def mark_complete(self, success=True):
        """Mark training as complete"""
        self.is_complete = True
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
        
        # View mode: 'detailed' or 'icons'
        self.view_mode = 'detailed'
        
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
        
        # Load directory
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
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # ML menu
        ml_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Machine Learning", menu=ml_menu)
        ml_menu.add_command(label="Model Configuration", command=self.open_model_config)
        ml_menu.add_command(label="Train from Folders", command=self.train_from_folders)
        ml_menu.add_command(label="Train from Tags", command=self.train_from_tags)
        ml_menu.add_command(label="Auto-Sort Images", command=self.auto_sort_images)
        ml_menu.add_command(label="Unsupervised Clustering", command=self.run_clustering)
        ml_menu.add_separator()
        ml_menu.add_command(label="View Metrics", command=self.show_metrics)
        
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
        
        ttk.Label(left_panel, text="Drives", font=('Arial', 10, 'bold')).pack(pady=5)
        
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
            stat = os.statvfs(drive)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used = total - free
            return total, used, free
    
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
        
        # Bind double-click and single-click event
        self.tree.bind('<Double-1>', self.on_tree_double_click)
        self.tree.bind('<ButtonRelease-1>', self.on_tree_single_click)
    
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
    
    def on_icons_configure(self, event):
        """Update scroll region when icons container changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def on_canvas_configure(self, event):
        """Update canvas window width to match canvas"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
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
            
            if self.view_mode == 'detailed':
                self.load_detailed_view(folders, files)
            else:
                self.load_icons_view(folders, files)
            
            # Update status
            self.status_var.set(f"{len(folders)} folders, {len(files)} files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load directory: {e}")
    
    def load_detailed_view(self, folders, files):
        """Load items in detailed view"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add folders
        for folder in folders:
            self.add_tree_item(folder, '📁 Folder')
        
        # Add files
        for file in files:
            self.add_tree_item(file, '📄 File')
    
    def load_icons_view(self, folders, files):
        """Load items in large icons view"""
        # Clear existing icon buttons
        for widget in self.icons_container.winfo_children():
            widget.destroy()
        self.icon_buttons = []
        self.image_references = []  # Clear image references
        
        # Combine folders and files
        all_items = folders + files
        
        # Create icon buttons in a grid
        col = 0
        row = 0
        max_cols = 5  # Number of icons per row
        
        for item in all_items:
            icon_frame = self.create_icon_widget(item)
            icon_frame.grid(row=row, column=col, padx=10, pady=10, sticky='n')
            self.icon_buttons.append((icon_frame, item))
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def create_icon_widget(self, path):
        """Create an icon widget for a file or folder"""
        frame = ttk.Frame(self.icons_container)
        
        # Check if it's an image file
        is_image = False
        if path.is_file():
            ext = path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico']:
                is_image = True
        
        # Try to load actual image thumbnail
        if is_image:
            try:
                # Load and resize image
                img = Image.open(path)
                
                # Convert RGBA to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Resize to thumbnail
                img.thumbnail((120, 120), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Store reference to prevent garbage collection
                self.image_references.append(photo)
                
                # Create label with actual image
                icon_label = tk.Label(frame, image=photo, bg='white')
                icon_label.image = photo  # Keep a reference
                icon_label.pack()
                
            except Exception as e:
                # If image loading fails, use emoji icon
                is_image = False
                print(f"Failed to load image {path.name}: {e}")
        
        # Use emoji icon for non-images or failed image loads
        if not is_image or 'icon_label' not in locals():
            # Icon label (using emoji as placeholder)
            if path.is_dir():
                icon_text = "📁"
            else:
                # Get file extension and assign icon
                ext = path.suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    icon_text = "🖼️"
                elif ext in ['.mp4', '.avi', '.mkv', '.mov']:
                    icon_text = "🎬"
                elif ext in ['.mp3', '.wav', '.flac', '.ogg']:
                    icon_text = "🎵"
                elif ext in ['.pdf']:
                    icon_text = "📕"
                elif ext in ['.doc', '.docx', '.txt']:
                    icon_text = "📄"
                elif ext in ['.zip', '.rar', '.7z']:
                    icon_text = "📦"
                else:
                    icon_text = "📄"
            
            icon_label = tk.Label(frame, text=icon_text, font=('Arial', 48), bg='white')
            icon_label.pack()
        
        # Name label (truncate if too long)
        name = path.name
        if len(name) > 15:
            name = name[:12] + "..."
        name_label = tk.Label(frame, text=name, font=('Arial', 9), bg='white', wraplength=120)
        name_label.pack()
        
        # Bind double-click and single-click
        icon_label.bind('<Double-1>', lambda e: self.on_icon_double_click(path))
        name_label.bind('<Double-1>', lambda e: self.on_icon_double_click(path))
        icon_label.bind('<Button-1>', lambda e: self.on_icon_single_click(path))
        name_label.bind('<Button-1>', lambda e: self.on_icon_single_click(path))
        
        return frame
    
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
            else:
                self.open_file(path)
    
    def add_tree_item(self, path, item_type):
        """Add an item to the tree view"""
        try:
            # Get file stats
            stats = path.stat()
            
            # Format size
            if path.is_file():
                size = self.format_size(stats.st_size)
            else:
                size = ""
            
            # Format creation and modification time
            from datetime import datetime
            created_time = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert into tree
            self.tree.insert('', tk.END, text=path.name, 
                           values=(item_type, size, created_time, mod_time),
                           tags=('folder' if path.is_dir() else 'file',))
            
        except Exception as e:
            # If we can't get stats, still add the item
            self.tree.insert('', tk.END, text=path.name, 
                           values=(item_type, 'N/A', 'N/A', 'N/A'),
                           tags=('folder' if path.is_dir() else 'file',))
    
    def format_size(self, size):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    
    def on_tree_single_click(self, event):
        """Handle single-click on tree item for image selection"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_text = self.tree.item(item, 'text')
        new_path = self.current_path / item_text
        
        if new_path.is_file():
            # Check if it's an image
            if new_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif']:
                self.on_image_selected(str(new_path))
    
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
            # For images, select them; for other files, open them
            if new_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif']:
                self.on_image_selected(str(new_path))
            else:
                self.open_file(new_path)
    
    def on_double_click(self, event):
        """Legacy handler for backward compatibility"""
        self.on_tree_double_click(event)
    
    def open_file(self, filepath):
        """Open file with default application"""
        try:
            if platform.system() == 'Windows':
                os.startfile(filepath)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{filepath}"')
            else:  # Linux
                os.system(f'xdg-open "{filepath}"')
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open file: {e}")
    
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
    
    def setup_tag_sidebar(self, parent):
        """Setup the tag management sidebar"""
        tag_sidebar = ttk.Frame(parent, width=250)
        tag_sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        tag_sidebar.pack_propagate(False)
        
        ttk.Label(tag_sidebar, text="Tag Management", font=('Arial', 10, 'bold')).pack(pady=5)
        
        # Selected image label
        self.selected_image_label = ttk.Label(tag_sidebar, text="No image selected", wraplength=240)
        self.selected_image_label.pack(pady=5)
        
        ttk.Separator(tag_sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Current tags section
        ttk.Label(tag_sidebar, text="Current Tags:", font=('Arial', 9, 'bold')).pack(anchor='w', padx=5)
        
        # Tags listbox
        tags_frame = ttk.Frame(tag_sidebar)
        tags_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tags_listbox = tk.Listbox(tags_frame, selectmode=tk.SINGLE, height=10)
        tags_scrollbar = ttk.Scrollbar(tags_frame, orient=tk.VERTICAL, command=self.tags_listbox.yview)
        self.tags_listbox.configure(yscrollcommand=tags_scrollbar.set)
        
        self.tags_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tags_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tag operations
        tag_ops_frame = ttk.Frame(tag_sidebar)
        tag_ops_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(tag_ops_frame, text="Remove Tag", command=self.remove_selected_tag).pack(fill=tk.X, pady=2)
        
        ttk.Separator(tag_sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Add tag section
        ttk.Label(tag_sidebar, text="Add Tag:", font=('Arial', 9, 'bold')).pack(anchor='w', padx=5)
        
        add_tag_frame = ttk.Frame(tag_sidebar)
        add_tag_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.new_tag_entry = ttk.Entry(add_tag_frame)
        self.new_tag_entry.pack(fill=tk.X, pady=2)
        
        ttk.Button(add_tag_frame, text="Add Manual Tag", command=self.add_manual_tag).pack(fill=tk.X, pady=2)
        
        ttk.Separator(tag_sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # ML predictions section
        ttk.Label(tag_sidebar, text="Predicted Tags:", font=('Arial', 9, 'bold')).pack(anchor='w', padx=5)
        
        predictions_frame = ttk.Frame(tag_sidebar)
        predictions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.predictions_listbox = tk.Listbox(predictions_frame, selectmode=tk.SINGLE, height=5)
        pred_scrollbar = ttk.Scrollbar(predictions_frame, orient=tk.VERTICAL, command=self.predictions_listbox.yview)
        self.predictions_listbox.configure(yscrollcommand=pred_scrollbar.set)
        
        self.predictions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pred_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(tag_sidebar, text="Apply Selected Prediction", command=self.apply_selected_prediction, 
                  ).pack(fill=tk.X, padx=5, pady=2)
    
    def initialize_ml_system(self):
        """Initialize the ML system in background"""
        def init_thread():
            try:
                self.controller.initialize(str(self.current_path))
                self.status_var.set("ML system ready")
            except Exception as e:
                self.status_var.set(f"ML initialization error: {str(e)[:50]}")
        
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
        tags = self.controller.tagging_engine.get_image_tags(self.selected_image_path)
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
        
        self.predictions_listbox.delete(0, tk.END)
        self.predictions_listbox.insert(tk.END, "Predicting...")
        
        def predict_thread():
            try:
                predictions = self.controller.tagging_engine.predict_tags_for_image(self.selected_image_path)
                
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
                
            except Exception as e:
                self.root.after(0, lambda: progress_callback('error', str(e)))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {e}"))
        
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()
    
    def auto_sort_images(self):
        """Auto-sort all images in current directory"""
        if not messagebox.askyesno("Confirm", 
            "This will predict tags for all images in the current directory. Continue?"):
            return
        
        self.status_var.set("Auto-sorting in progress...")
        
        def sort_thread():
            try:
                results = self.controller.start_initial_sorting(auto_apply=True)
                total = len(results)
                self.root.after(0, lambda: messagebox.showinfo("Complete", 
                    f"Auto-sorting completed for {total} images"))
                self.root.after(0, lambda: self.status_var.set(f"Sorted {total} images"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Auto-sorting failed: {e}"))
        
        thread = threading.Thread(target=sort_thread, daemon=True)
        thread.start()
    
    def run_clustering(self):
        """Run unsupervised clustering"""
        n_clusters = tk.simpledialog.askinteger("Clustering", 
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
    root = tk.Tk()
    app = FileExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
