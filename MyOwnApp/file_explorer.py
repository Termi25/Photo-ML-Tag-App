"""
Basic File Explorer
A simple file explorer application using tkinter
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, Canvas
from pathlib import Path
import platform
import string
from PIL import Image, ImageTk


class FileExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("File Explorer")
        self.root.geometry("1100x700")
        
        # Current directory
        self.current_path = Path.home()
        
        # View mode: 'detailed' or 'icons'
        self.view_mode = 'detailed'
        
        # Store image references to prevent garbage collection
        self.image_references = []
        
        # Setup UI
        self.setup_ui()
        self.load_directory(self.current_path)
        
    def setup_ui(self):
        """Setup the user interface"""
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
        self.icons_btn = ttk.Button(toolbar, text="🖼️ Large Icons", command=self.switch_to_icons)
        self.icons_btn.pack(side=tk.LEFT, padx=2)
        
        # Address bar
        ttk.Label(toolbar, text="Path:").pack(side=tk.LEFT, padx=(10, 2))
        self.path_var = tk.StringVar(value=str(self.current_path))
        self.path_entry = ttk.Entry(toolbar, textvariable=self.path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.path_entry.bind('<Return>', lambda e: self.navigate_to_path())
        
        # Go button
        self.go_btn = ttk.Button(toolbar, text="Go", command=self.navigate_to_path)
        self.go_btn.pack(side=tk.LEFT, padx=2)
        
        # Main container with left panel and content area
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
        
        # Right panel - content area
        self.content_frame = ttk.Frame(main_container)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create both views
        self.create_detailed_view()
        self.create_icons_view()
        
        # Show detailed view by default
        self.show_detailed_view()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
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
        
        # Bind double-click event
        self.tree.bind('<Double-1>', self.on_tree_double_click)
    
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
        
        # Bind double-click
        icon_label.bind('<Double-1>', lambda e: self.on_icon_double_click(path))
        name_label.bind('<Double-1>', lambda e: self.on_icon_double_click(path))
        
        return frame
    
    def on_icon_double_click(self, path):
        """Handle double-click on icon"""
        if path.is_dir():
            self.history.append(self.current_path)
            self.load_directory(path)
        elif path.is_file():
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
            # Open file with default application
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


def main():
    """Main entry point"""
    root = tk.Tk()
    app = FileExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
