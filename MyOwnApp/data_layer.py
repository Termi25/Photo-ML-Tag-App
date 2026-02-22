"""
Data Layer
Handles persistent storage including image metadata (tags), model weights, and database operations.
Decouples physical file location from semantic identity.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import pickle


class MetadataStore:
    """SQLite-based metadata storage for image tags and properties"""
    
    def __init__(self, db_path: str = "./data/image_metadata.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
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
        
        self.connection.commit()
        print(f"Database initialized at {self.db_path}")
    
    def add_image(self, file_path: str, file_name: str, file_size: int = 0) -> int:
        """Add an image to the database"""
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        
        try:
            cursor.execute('''
                INSERT INTO images (file_path, file_name, file_size, added_to_db)
                VALUES (?, ?, ?, ?)
            ''', (file_path, file_name, file_size, now))
            self.connection.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Image already exists, return its ID
            cursor.execute('SELECT id FROM images WHERE file_path = ?', (file_path,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_or_create_tag(self, tag_name: str, category: str = None, color: str = None) -> int:
        """Get existing tag or create new one"""
        cursor = self.connection.cursor()
        
        # Try to get existing tag
        cursor.execute('SELECT id FROM tags WHERE tag_name = ?', (tag_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Create new tag
        now = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO tags (tag_name, category, color, created_at)
            VALUES (?, ?, ?, ?)
        ''', (tag_name, category, color, now))
        self.connection.commit()
        return cursor.lastrowid
    
    def add_image_tag(self, image_id: int, tag_id: int, confidence: float = 1.0, source: str = 'manual'):
        """Associate a tag with an image"""
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        
        try:
            cursor.execute('''
                INSERT INTO image_tags (image_id, tag_id, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_id, tag_id, confidence, source, now))
            self.connection.commit()
        except sqlite3.IntegrityError:
            # Update existing association
            cursor.execute('''
                UPDATE image_tags 
                SET confidence = ?, source = ?, created_at = ?
                WHERE image_id = ? AND tag_id = ?
            ''', (confidence, source, now, image_id, tag_id))
            self.connection.commit()
    
    def get_image_tags(self, image_id: int) -> List[Dict[str, Any]]:
        """Get all tags for an image"""
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
        return results
    
    def get_image_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get image record by file path"""
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM images WHERE file_path = ?', (file_path,))
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'file_path': row[1],
                'file_name': row[2],
                'file_size': row[3],
                'created_at': row[4],
                'modified_at': row[5],
                'added_to_db': row[6],
                'last_processed': row[7]
            }
        return None
    
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
        """Get all tags in the system"""
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
        return results
    
    def remove_image_tag(self, image_id: int, tag_id: int):
        """Remove a tag from an image"""
        cursor = self.connection.cursor()
        cursor.execute('DELETE FROM image_tags WHERE image_id = ? AND tag_id = ?', (image_id, tag_id))
        self.connection.commit()
    
    def update_last_processed(self, image_id: int):
        """Update the last processed timestamp"""
        cursor = self.connection.cursor()
        now = datetime.now().isoformat()
        cursor.execute('UPDATE images SET last_processed = ? WHERE id = ?', (now, image_id))
        self.connection.commit()
    
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
    """Handles saving and loading of ML model weights"""
    
    def __init__(self, models_dir: str = "./data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_weights(self, model_state: Any, model_name: str, metadata: Dict[str, Any] = None):
        """Save model weights and metadata"""
        model_path = self.models_dir / f"{model_name}_weights.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        # Save model weights (using pickle for flexibility with different frameworks)
        with open(model_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['model_name'] = model_name
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved: {model_path}")
    
    def load_model_weights(self, model_name: str) -> tuple:
        """Load model weights and metadata"""
        model_path = self.models_dir / f"{model_name}_weights.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load weights
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model_state, metadata
    
    def list_models(self) -> List[str]:
        """List all available models"""
        models = []
        for file in self.models_dir.glob("*_weights.pkl"):
            model_name = file.stem.replace('_weights', '')
            models.append(model_name)
        return models
    
    def model_exists(self, model_name: str) -> bool:
        """Check if model exists"""
        model_path = self.models_dir / f"{model_name}_weights.pkl"
        return model_path.exists()
    
    def delete_model(self, model_name: str):
        """Delete a model and its metadata"""
        model_path = self.models_dir / f"{model_name}_weights.pkl"
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        
        if model_path.exists():
            model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        print(f"Model deleted: {model_name}")


# Initialize global instances
metadata_store = MetadataStore()
model_storage = ModelStorage()
