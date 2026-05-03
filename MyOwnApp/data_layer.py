"""
Data Layer
Handles persistent storage including image metadata (tags), model weights, and database operations.
Decouples physical file location from semantic identity.

Redis is used as an optional cache layer for faster repeated lookups.
If Redis is not available, falls back to an in-memory LRU-style cache.
"""

import sqlite3
import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import pickle

# Sentinel for "value not in cache" — distinct from None (valid return)
_CACHE_MISS = object()

_CACHE_MAX = 8192  # max entries in the in-process LRU cache


class TagCache:
    """In-process LRU cache for image/tag lookups.

    Uses a single ``collections.OrderedDict`` so that get, set, and eviction
    are all O(1).  No network dependency, no serialisation overhead.
    """

    def __init__(self, max_size: int = _CACHE_MAX) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max = max_size

    def get(self, key: str) -> Any:
        try:
            self._store.move_to_end(key)
            return self._store[key]
        except KeyError:
            return None

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
        else:
            if len(self._store) >= self._max:
                self._store.popitem(last=False)  # evict oldest (FIFO end)
            self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def delete_pattern(self, prefix: str) -> None:
        for k in [k for k in self._store if k.startswith(prefix)]:
            del self._store[k]

    def flush(self) -> None:
        self._store.clear()

    @property
    def backend(self) -> str:
        return "memory"


# Module-level shared cache instance
_tag_cache = TagCache()


class MetadataStore:
    """SQLite-based metadata storage for image tags and properties,
    with an optional Redis/in-memory cache for high-frequency reads."""
    
    def __init__(self, db_path: str = "./data/image_metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: sqlite3.Connection
        self._cache = _tag_cache
        # Per-folder cache of hidden image paths — invalidated on set_image_hidden()
        self._hidden_folder_cache: Dict[str, set] = {}
        self.init_database()
    
    # ------------------------------------------------------------------
    # Root-folder database management
    # ------------------------------------------------------------------

    def set_root_folder(self, root_folder: str) -> None:
        """Place (or reuse) the single shared database at the *root_folder* level.

        Call this once when the user opens a root folder via settings.
        Navigating into sub-folders does NOT move the database.
        """
        if self.connection:
            self.connection.close()

        root_path = Path(root_folder)
        self.db_path = root_path / ".image_tags.db"
        root_path.mkdir(parents=True, exist_ok=True)
        self._cache.flush()              # wipe tag cache on DB switch
        self._hidden_folder_cache = {}   # wipe hidden-paths cache
        self.init_database()
        self._hide_database_file()
        print(f"Database root set to: {self.db_path}")

    def change_database_location(self, directory: str) -> None:
        """Legacy helper — kept for API compatibility but now a no-op when
        *directory* is just a sub-folder navigation.  The database stays in
        whichever root folder was last set via ``set_root_folder``."""
        # Only actually move the DB if the caller is explicitly setting a new
        # root (i.e. this is the same path as our current db_path parent or a
        # completely different root).  For ordinary sub-folder navigation the
        # load_directory code should NOT call this anymore.
        pass
    
    def _hide_database_file(self):
        """Hide the database file on Windows"""
        import platform
        if platform.system() == 'Windows' and self.db_path.exists():
            try:
                import ctypes
                # Set FILE_ATTRIBUTE_HIDDEN (0x02)
                ctypes.windll.kernel32.SetFileAttributesW(str(self.db_path), 0x02)
            except Exception as e:
                # Silently fail if we can't hide the file
                pass
    
    def init_database(self):
        """Initialize database schema"""
        # Check if database is new
        is_new_db = not self.db_path.exists()
        
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # WAL mode: allows concurrent reads while a write is in progress,
        # preventing the background training thread from blocking the UI thread.
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")  # faster fsync, still safe with WAL
        cursor = self.connection.cursor()
        
        # Images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                created_at TEXT,
                modified_at TEXT,
                added_to_db TEXT,
                last_processed TEXT
            )
        ''')
        
        # Tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT UNIQUE NOT NULL,
                category TEXT,
                color TEXT,
                created_at TEXT
            )
        ''')
        
        # Image-Tag associations (many-to-many)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                created_at TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                UNIQUE(image_id, tag_id)
            )
        ''')
        
        # Ground truth for validation/testing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ground_truth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                verified_by TEXT,
                verified_at TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
                UNIQUE(image_id, tag_id)
            )
        ''')
        
        # Model training history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                training_start TEXT,
                training_end TEXT,
                epochs INTEGER,
                final_accuracy REAL,
                hyperparameters TEXT,
                notes TEXT
            )
        ''')

        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_path ON images(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag_name ON tags(tag_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_tag ON image_tags(image_id, tag_id)')

        # Migration: add 'hidden' column if it was not present in older databases
        try:
            cursor.execute('ALTER TABLE images ADD COLUMN hidden INTEGER DEFAULT 0')
            self.connection.commit()
        except Exception:
            pass  # Column already exists

        # Index that makes WHERE hidden = 1 fast even on large collections
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_hidden ON images(hidden, file_path)')

        self.connection.commit()
        
        # Hide the database file on Windows to prevent SQLite Browser from opening it
        self._hide_database_file()
        
        print(f"Database initialized at {self.db_path}")
    
    def add_image(self, file_path: str, file_name: str, file_size: int = 0,
                  commit: bool = True) -> int | None:
        """Add an image to the database (normalizes path for consistency).

        Pass ``commit=False`` when inserting inside a larger batch transaction
        managed by the caller; remember to call ``connection.commit()`` afterwards.
        """
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        
        # Normalize path to absolute path for consistency
        try:
            normalized_path = str(Path(file_path).resolve())
        except:
            normalized_path = file_path
        
        try:
            cursor.execute('''
                INSERT INTO images (file_path, file_name, file_size, added_to_db)
                VALUES (?, ?, ?, ?)
            ''', (normalized_path, file_name, file_size, now))
            if commit:
                self.connection.commit()
            image_id = cursor.lastrowid
            # Invalidate any stale cached record for this path
            self._cache.delete(f"img:{normalized_path}")
            self._cache.delete(f"img:{file_path}")
            return image_id
        except sqlite3.IntegrityError:
            # Image already exists, return its ID
            cursor.execute('SELECT id FROM images WHERE file_path = ?', (normalized_path,))
            result = cursor.fetchone()
            if result:
                return result[0]
            
            # Try fallback with original path in case it was stored differently
            cursor.execute('SELECT id FROM images WHERE file_path = ?', (file_path,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_or_create_tag(self, tag_name: str, category: str | None = None,
                           color: str | None = None, commit: bool = True) -> int:
        """Get existing tag or create new one.

        Pass ``commit=False`` inside a batch transaction managed by the caller.
        """
        cache_key = f"tag_id:{tag_name}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        cursor = self.connection.cursor()
        
        # Try to get existing tag
        cursor.execute('SELECT id FROM tags WHERE tag_name = ?', (tag_name,))
        result = cursor.fetchone()
        
        if result:
            self._cache.set(cache_key, result[0])
            return result[0]
        
        # Create new tag
        now = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO tags (tag_name, category, color, created_at)
            VALUES (?, ?, ?, ?)
        ''', (tag_name, category, color, now))
        if commit:
            self.connection.commit()
        tag_id = cursor.lastrowid
        if tag_id is None:
            # Fallback: query for the just-created tag
            cursor.execute('SELECT id FROM tags WHERE tag_name = ?', (tag_name,))
            result = cursor.fetchone()
            tag_id = result[0] if result else 0
        # Invalidate the all-tags list cache
        self._cache.delete('all_tags')
        self._cache.set(cache_key, tag_id)
        return tag_id
    
    def add_image_tag(self, image_id: int, tag_id: int, confidence: float = 1.0,
                      source: str = 'manual', commit: bool = True):
        """Associate a tag with an image.

        Pass ``commit=False`` inside a batch transaction managed by the caller.
        """
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        
        try:
            cursor.execute('''
                INSERT INTO image_tags (image_id, tag_id, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_id, tag_id, confidence, source, now))
            if commit:
                self.connection.commit()
        except sqlite3.IntegrityError:
            # Update existing association
            cursor.execute('''
                UPDATE image_tags 
                SET confidence = ?, source = ?, created_at = ?
                WHERE image_id = ? AND tag_id = ?
            ''', (confidence, source, now, image_id, tag_id))
            if commit:
                self.connection.commit()
        # Invalidate cached tag list for this image
        self._cache.delete(f"image_tags:{image_id}")
    
    def get_image_tags(self, image_id: int) -> List[Dict[str, Any]]:
        """Get all tags for an image (cached)"""
        cache_key = f"image_tags:{image_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT t.id, t.tag_name, t.category, t.color, it.confidence, it.source
            FROM tags t
            JOIN image_tags it ON t.id = it.tag_id
            WHERE it.image_id = ?
            ORDER BY it.confidence DESC
        ''', (image_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'tag_name': row[1],
                'category': row[2],
                'color': row[3],
                'confidence': row[4],
                'source': row[5]
            })
        self._cache.set(cache_key, results)
        return results
    
    def get_image_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get image record by file path (normalized, cached)"""
        try:
            normalized_input = str(Path(file_path).resolve())
        except Exception:
            normalized_input = file_path

        cache_key = f"img:{normalized_input}"
        cached = self._cache.get(cache_key)
        if cached is not _CACHE_MISS and cached is not None:
            return cached  # type: ignore[return-value]
        if cached is _CACHE_MISS:
            return None

        cursor = self.connection.cursor()

        # Use the normalized path — add_image() stores resolved paths
        cursor.execute('SELECT * FROM images WHERE file_path = ?', (normalized_input,))
        row = cursor.fetchone()

        if row:
            record = {
                'id': row[0],
                'file_path': row[1],
                'file_name': row[2],
                'file_size': row[3],
                'created_at': row[4],
                'modified_at': row[5],
                'added_to_db': row[6],
                'last_processed': row[7]
            }
            self._cache.set(cache_key, record)
            return record
        # Cache miss — store sentinel so we skip the DB on repeated lookups
        self._cache.set(cache_key, _CACHE_MISS)
        return None
    
    def search_images_advanced(
        self,
        included_tags: List[str],
        excluded_tags: List[str],
        sort_by: str = 'name',
        sort_asc: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return images that have ALL included_tags and NONE of excluded_tags.

        Parameters
        ----------
        included_tags : tags the image must have (AND semantics).
        excluded_tags : tags the image must NOT have.
        sort_by       : 'name' | 'date' | 'size'
        sort_asc      : ascending when True.
        """
        cursor = self.connection.cursor()

        # Build the ORDER BY clause
        _order_col = {
            'name': 'i.file_name',
            'date': 'i.added_to_db',
            'size': 'i.file_size',
        }.get(sort_by, 'i.file_name')
        _direction = 'ASC' if sort_asc else 'DESC'

        params: list = []

        # Included-tags subquery (AND — image must have every included tag)
        if included_tags:
            incl_placeholders = ','.join('?' * len(included_tags))
            included_clause = f"""
                i.id IN (
                    SELECT it.image_id
                    FROM image_tags it JOIN tags t ON it.tag_id = t.id
                    WHERE t.tag_name IN ({incl_placeholders})
                    GROUP BY it.image_id
                    HAVING COUNT(DISTINCT t.tag_name) = ?
                )"""
            params += included_tags + [len(included_tags)]
        else:
            included_clause = '1=1'

        # Excluded-tags subquery (image must NOT have any excluded tag)
        if excluded_tags:
            excl_placeholders = ','.join('?' * len(excluded_tags))
            excluded_clause = f"""
                i.id NOT IN (
                    SELECT it.image_id
                    FROM image_tags it JOIN tags t ON it.tag_id = t.id
                    WHERE t.tag_name IN ({excl_placeholders})
                )"""
            params += excluded_tags
        else:
            excluded_clause = '1=1'

        query = f"""
            SELECT i.id, i.file_path, i.file_name,
                   GROUP_CONCAT(t.tag_name) AS tags
            FROM images i
            LEFT JOIN image_tags it ON i.id = it.image_id
            LEFT JOIN tags t ON it.tag_id = t.id
            WHERE {included_clause}
              AND {excluded_clause}
            GROUP BY i.id
            ORDER BY {_order_col} {_direction}
        """
        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                'id':        row[0],
                'file_path': row[1],
                'file_name': row[2],
                'tags':      row[3].split(',') if row[3] else [],
            })
        return results

    def search_images_by_tags(self, tag_names: List[str], match_all: bool = True) -> List[Dict[str, Any]]:
        """Search images by tags with Boolean logic"""
        cursor = self.connection.cursor()
        
        if match_all:
            # AND logic - image must have all tags
            query = '''
                SELECT i.*, GROUP_CONCAT(t.tag_name) as tags
                FROM images i
                JOIN image_tags it ON i.id = it.image_id
                JOIN tags t ON it.tag_id = t.id
                WHERE t.tag_name IN ({})
                GROUP BY i.id
                HAVING COUNT(DISTINCT t.tag_name) = ?
            '''.format(','.join('?' * len(tag_names)))
            cursor.execute(query, tag_names + [len(tag_names)])
        else:
            # OR logic - image must have at least one tag
            query = '''
                SELECT DISTINCT i.*, GROUP_CONCAT(t.tag_name) as tags
                FROM images i
                JOIN image_tags it ON i.id = it.image_id
                JOIN tags t ON it.tag_id = t.id
                WHERE t.tag_name IN ({})
                GROUP BY i.id
            '''.format(','.join('?' * len(tag_names)))
            cursor.execute(query, tag_names)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'file_path': row[1],
                'file_name': row[2],
                'tags': row[-1].split(',') if row[-1] else []
            })
        return results
    
    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags in the system (cached)"""
        cached = self._cache.get('all_tags')
        if cached is not None:
            return cached

        cursor = self.connection.cursor()
        cursor.execute('SELECT id, tag_name, category, color FROM tags ORDER BY tag_name')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'tag_name': row[1],
                'category': row[2],
                'color': row[3]
            })
        self._cache.set('all_tags', results)
        return results
    
    def remove_image_tag(self, image_id: int, tag_id: int):
        """Remove a tag from an image"""
        cursor = self.connection.cursor()
        cursor.execute('DELETE FROM image_tags WHERE image_id = ? AND tag_id = ?', (image_id, tag_id))
        self.connection.commit()
        self._cache.delete(f"image_tags:{image_id}")
    
    def update_last_processed(self, image_id: int):
        """Update the last processed timestamp"""
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        cursor.execute('UPDATE images SET last_processed = ? WHERE id = ?', (now, image_id))
        self.connection.commit()

    # ------------------------------------------------------------------
    # Hidden-image management
    # ------------------------------------------------------------------

    def set_image_hidden(self, image_id: int, hidden: bool) -> None:
        """Mark or unmark an image as hidden inside the app database."""
        cursor = self.connection.cursor()
        cursor.execute('UPDATE images SET hidden = ? WHERE id = ?',
                       (1 if hidden else 0, image_id))
        self.connection.commit()
        # Invalidate tag-record cache
        cursor.execute('SELECT file_path FROM images WHERE id = ?', (image_id,))
        row = cursor.fetchone()
        if row:
            self._cache.delete(f"img:{row[0]}")
            try:
                self._cache.delete(f"img:{Path(row[0]).resolve()}")
            except Exception:
                pass
        # Invalidate the folder hidden-paths cache (any folder may be affected)
        self._hidden_folder_cache.clear()

    def is_image_hidden(self, image_id: int) -> bool:
        """Return True if the image is marked hidden in the app database."""
        cursor = self.connection.cursor()
        cursor.execute('SELECT hidden FROM images WHERE id = ?', (image_id,))
        row = cursor.fetchone()
        return bool(row[0]) if (row and row[0]) else False

    def get_hidden_image_paths_in_folder(self, folder: str) -> set:
        """Return the set of absolute file paths that are marked hidden
        and reside inside *folder*.

        Results are cached per folder and invalidated whenever set_image_hidden()
        is called, so repeated calls during a single directory view are free.
        The idx_images_hidden composite index (hidden, file_path) makes the DB
        scan cheap even for large collections.
        """
        try:
            folder_resolved = str(Path(folder).resolve())
        except Exception:
            folder_resolved = folder

        cached = self._hidden_folder_cache.get(folder_resolved)
        if cached is not None:
            return cached

        cursor = self.connection.cursor()
        # The composite index on (hidden, file_path) turns this into an index range scan
        cursor.execute('SELECT file_path FROM images WHERE hidden = 1')
        result: set = set()
        for (fp,) in cursor.fetchall():
            try:
                fp_resolved = str(Path(fp).resolve())
                if fp_resolved.startswith(folder_resolved):
                    result.add(fp_resolved)
            except Exception:
                pass

        self._hidden_folder_cache[folder_resolved] = result
        return result

    # ------------------------------------------------------------------
    # Orphan database cleanup
    # ------------------------------------------------------------------

    def clean_orphan_databases(self, root_folder: str) -> List[str]:
        """Delete every ``.image_tags.db`` found under *root_folder* that is
        **not** the currently active database.  Also removes the built-in
        default ``./data/image_metadata.db`` if it differs from the active DB.

        Returns a list of removed file paths.
        """
        current_db = self.db_path.resolve()
        removed: List[str] = []

        # Walk the root tree looking for stale .image_tags.db files
        for db_file in Path(root_folder).rglob('.image_tags.db'):
            if db_file.resolve() != current_db:
                try:
                    db_file.unlink()
                    removed.append(str(db_file))
                    print(f"Removed orphan database: {db_file}")
                except Exception as exc:
                    print(f"Could not remove {db_file}: {exc}")

        # Optionally remove the original default DB in ./data/
        default_db = (Path(__file__).parent / 'data' / 'image_metadata.db').resolve()
        if default_db != current_db and default_db.exists():
            try:
                default_db.unlink()
                removed.append(str(default_db))
                print(f"Removed default orphan database: {default_db}")
            except Exception as exc:
                print(f"Could not remove default DB: {exc}")

        return removed

    def add_ground_truth(self, image_id: int, tag_id: int, verified_by: str = 'user'):
        """Add ground truth annotation"""
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        
        try:
            cursor.execute('''
                INSERT INTO ground_truth (image_id, tag_id, verified_by, verified_at)
                VALUES (?, ?, ?, ?)
            ''', (image_id, tag_id, verified_by, now))
            self.connection.commit()
        except sqlite3.IntegrityError:
            pass  # Already exists
    
    def get_ground_truth_tags(self, image_id: int) -> Set[int]:
        """Get ground truth tags for an image"""
        cursor = self.connection.cursor()
        cursor.execute('SELECT tag_id FROM ground_truth WHERE image_id = ?', (image_id,))
        return {row[0] for row in cursor.fetchall()}
    
    def log_training_session(self, model_name: str, epochs: int, final_accuracy: float, 
                            hyperparameters: Dict[str, Any], notes: str = ""):
        """Log a training session"""
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO training_history 
            (model_name, training_start, training_end, epochs, final_accuracy, hyperparameters, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (model_name, now, now, epochs, final_accuracy, json.dumps(hyperparameters), notes))
        self.connection.commit()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class ModelStorage:
    """Handles saving and loading of ML model weights.

    Directory layout (rooted at the user's chosen root folder)::

        <root>/
          ML_Training_Data/
            model_folder/          ← pickle weights + metadata JSON
            data_training_graphs/  ← accuracy/loss PNG + training JSON stats

    File naming convention:
        model_<type>_<YYYYMMDD_HHMMSS>.pkl
        model_<type>_<YYYYMMDD_HHMMSS>_metadata.json
        training_<type>_<YYYYMMDD_HHMMSS>.json
        graph_<graphtype>_<YYYYMMDD_HHMMSS>.png

    A ``latest_weights.pkl`` / ``latest_metadata.json`` pair is also kept so
    that InferenceEngine.load_model('latest') continues to work without scanning.
    """

    _ML_ROOT        = 'ML_Training_Data'
    _MODEL_FOLDER   = 'model_folder'
    _GRAPH_FOLDER   = 'data_training_graphs'

    def __init__(self, models_dir: str = "./data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # graphs / stats land next to models by default until a root is set
        self.graphs_dir: Path = self.models_dir.parent / 'training_graphs'
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def set_root_folder(self, root_folder: str) -> None:
        """Point both storage directories at the ML_Training_Data subtree."""
        root = Path(root_folder)
        ml_root = root / self._ML_ROOT
        self.models_dir = ml_root / self._MODEL_FOLDER
        self.graphs_dir = ml_root / self._GRAPH_FOLDER
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model storage  : {self.models_dir}")
        print(f"Graph storage  : {self.graphs_dir}")

    # ------------------------------------------------------------------

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_model_weights(self, model_state: Any, model_name: str,
                           metadata: Dict[str, Any] | None = None):
        """Save model weights and metadata.

        Writes:
        * ``<model_name>_weights.pkl`` + ``<model_name>_metadata.json``  (legacy / auto-load)
        * ``model_<type>_<ts>.pkl``    + ``model_<type>_<ts>_metadata.json``  (timestamped)
        """
        if metadata is None:
            metadata = {}
        metadata['saved_at']   = datetime.now().isoformat()
        metadata['model_name'] = model_name

        ts          = self._ts()
        model_type  = metadata.get('model_type', model_name)
        data_source = metadata.get('data_source', '')
        stamp_tag   = f"{model_type}_{data_source}" if data_source else model_type

        # ---- legacy "latest" files kept for InferenceEngine.load_model('latest') ----
        legacy_weights  = self.models_dir / f"{model_name}_weights.pkl"
        legacy_metadata = self.models_dir / f"{model_name}_metadata.json"

        # Ensure any torch tensors are moved to CPU before pickling to avoid
        # issues when tensors reside on CUDA devices (which can hang or
        # produce large GPU-memory references during pickling).
        try:
            import torch

            def _to_cpu(obj):
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                if isinstance(obj, torch.Tensor):
                    return obj.cpu()
                return obj

            model_state_to_save = _to_cpu(model_state)
        except Exception:
            model_state_to_save = model_state

        with open(legacy_weights, 'wb') as f:
            pickle.dump(model_state_to_save, f)
        with open(legacy_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)

        # ---- timestamped copy ----
        stamp_weights  = self.models_dir / f"model_{stamp_tag}_{ts}.pkl"
        stamp_metadata = self.models_dir / f"model_{stamp_tag}_{ts}_metadata.json"
        with open(stamp_weights, 'wb') as f:
            pickle.dump(model_state_to_save if 'model_state_to_save' in locals() else model_state, f)
        with open(stamp_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved           : {legacy_weights}")
        print(f"Model saved (stamped) : {stamp_weights}")

    def save_training_stats(self, stats: Dict[str, Any], model_type: str) -> Path:
        """Persist training statistics JSON to ``training_<type>_<ts>.json``."""
        ts   = self._ts()
        path = self.graphs_dir / f"training_{model_type}_{ts}.json"
        payload = dict(stats)
        payload['saved_at'] = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Training stats saved  : {path}")
        return path

    def save_test_results(self, results: Dict[str, Any], model_type: str) -> Path:
        """Persist held-out test-set evaluation results to ``test_results_<type>_<ts>.json``."""
        ts   = self._ts()
        path = self.graphs_dir / f"test_results_{model_type}_{ts}.json"
        payload = dict(results)
        payload['saved_at']   = datetime.now().isoformat()
        payload['model_type'] = model_type
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Test results saved    : {path}")
        return path

    def save_training_graph(self, fig: Any, graph_type: str, model_type: str) -> Path:
        """Save a matplotlib Figure as ``graph_<type>_<ts>.png``."""
        ts   = self._ts()
        path = self.graphs_dir / f"graph_{graph_type}_{ts}.png"
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        print(f"Graph saved           : {path}")
        return path

    def load_model_weights(self, model_name: str) -> tuple:
        """Load model weights and metadata"""
        # Support both legacy names (<name>_weights.pkl) and timestamped
        # names (model_<type>_<YYYYMMDD_HHMMSS>.pkl).
        name = str(model_name).strip()
        if name.endswith('.pkl'):
            name = Path(name).stem

        legacy_model_path    = self.models_dir / f"{name}_weights.pkl"
        legacy_metadata_path = self.models_dir / f"{name}_metadata.json"

        stamped_model_path    = self.models_dir / f"{name}.pkl"
        stamped_metadata_path = self.models_dir / f"{name}_metadata.json"

        if legacy_model_path.exists():
            model_path = legacy_model_path
            metadata_path = legacy_metadata_path
        elif stamped_model_path.exists():
            model_path = stamped_model_path
            metadata_path = stamped_metadata_path
        else:
            raise FileNotFoundError(f"Model not found: {legacy_model_path} or {stamped_model_path}")

        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)

        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return model_state, metadata

    def list_models(self) -> List[str]:
        """List all available models"""
        # Prefer showing the most recently written models first.
        found: Dict[str, float] = {}

        # Legacy: <name>_weights.pkl
        for file in self.models_dir.glob("*_weights.pkl"):
            try:
                model_name = file.stem.replace('_weights', '')
                found[model_name] = max(found.get(model_name, 0.0), file.stat().st_mtime)
            except Exception:
                pass

        # Timestamped: model_<type>_<ts>.pkl (saved alongside metadata)
        for file in self.models_dir.glob("model_*.pkl"):
            try:
                model_name = file.stem
                found[model_name] = max(found.get(model_name, 0.0), file.stat().st_mtime)
            except Exception:
                pass

        models = sorted(found.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ts in models]

    def model_exists(self, model_name: str) -> bool:
        """Check if model exists"""
        name = str(model_name).strip()
        if name.endswith('.pkl'):
            name = Path(name).stem
        return (
            (self.models_dir / f"{name}_weights.pkl").exists()
            or (self.models_dir / f"{name}.pkl").exists()
        )

    def delete_model(self, model_name: str):
        """Delete a model and its metadata"""
        name = str(model_name).strip()
        if name.endswith('.pkl'):
            name = Path(name).stem

        # Try legacy
        legacy_model_path    = self.models_dir / f"{name}_weights.pkl"
        legacy_metadata_path = self.models_dir / f"{name}_metadata.json"
        # Try timestamped
        stamped_model_path    = self.models_dir / f"{name}.pkl"
        stamped_metadata_path = self.models_dir / f"{name}_metadata.json"

        removed_any = False
        for p in (legacy_model_path, legacy_metadata_path, stamped_model_path, stamped_metadata_path):
            try:
                if p.exists():
                    p.unlink()
                    removed_any = True
            except Exception:
                pass

        if removed_any:
            print(f"Model deleted: {name}")
        else:
            print(f"Model not found for deletion: {name}")


# Initialize global instances
metadata_store = MetadataStore()
model_storage = ModelStorage()
