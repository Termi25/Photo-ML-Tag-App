"""
Logic Controller
Orchestration layer that coordinates File Watcher, Tagging Engine, and Training Manager.
Acts as the bridge between UI and ML Module.
"""

import os
import re
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from PIL import Image
import piexif

from config import config_manager
from data_layer import metadata_store, model_storage
from ml_module import InferenceEngine, TrainingLoop, UnsupervisedClustering, ImagePreprocessor


class FileWatcher:
    """Monitors directories for image files and tracks changes"""
    
    def __init__(self):
        self.observer = None
        self.watched_paths = set()
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif', '.tiff'}
        self.on_new_images_callback = None
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """Scan directory for image files"""
        image_files = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"Directory does not exist: {directory}")
            return []
        
        if recursive:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                    image_files.append(str(file_path))
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                    image_files.append(str(file_path))
        
        print(f"Found {len(image_files)} images in {directory}")
        return image_files
    
    def register_images_to_database(self, image_paths: List[str]):
        """Register discovered images in the database"""
        for img_path in image_paths:
            path_obj = Path(img_path)
            file_size = path_obj.stat().st_size if path_obj.exists() else 0
            
            # Add to database if not already present
            image_id = metadata_store.add_image(
                file_path=str(img_path),
                file_name=path_obj.name,
                file_size=file_size
            )
        
        print(f"Registered {len(image_paths)} images to database")
    
    def watch_directory(self, directory: str, callback: Callable | None = None):
        """Start watching a directory for changes"""
        if self.observer is not None:
            self.stop_watching()
        
        self.on_new_images_callback = callback
        
        event_handler = ImageFileEventHandler(self.image_extensions, callback)
        self.observer = Observer()
        self.observer.schedule(event_handler, directory, recursive=True)
        self.observer.start()
        self.watched_paths.add(directory)
        
        print(f"Watching directory: {directory}")
    
    def stop_watching(self):
        """Stop watching directories"""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.watched_paths.clear()
            print("Stopped watching directories")


class ImageFileEventHandler(FileSystemEventHandler):
    """Handle file system events for image files"""
    
    def __init__(self, image_extensions: set, callback: Callable | None = None):
        self.image_extensions = image_extensions
        self.callback = callback
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(str(event.src_path))
            if file_path.suffix.lower() in self.image_extensions:
                print(f"New image detected: {file_path}")
                if self.callback:
                    self.callback([str(file_path)])


class TaggingEngine:
    """Manages image tag operations and predictions"""
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        self.model_loaded = False
    
    def load_model(self, model_name: str = 'latest') -> bool:
        """Load inference model"""
        self.model_loaded = self.inference_engine.load_model(model_name)
        return self.model_loaded
    
    def predict_tags_for_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Predict tags for a single image"""
        if not self.model_loaded:
            print("No model loaded for inference")
            return []
        
        predictions = self.inference_engine.predict_single(image_path)
        return predictions
    
    def predict_tags_batch(self, image_paths: List[str], progress_callback: Callable | None = None) -> Dict[str, List[Dict[str, Any]]]:
        """Predict tags for multiple images with progress feedback"""
        if not self.model_loaded:
            print("⚠ No model loaded for inference")
            if progress_callback:
                progress_callback('error', 'No model loaded')
            return {path: [] for path in image_paths}
        
        print(f"Starting batch prediction for {len(image_paths)} images...")
        if progress_callback:
            progress_callback('status', f'Predicting tags for {len(image_paths)} images...')
        
        predictions_list = self.inference_engine.predict_batch(image_paths)
        predictions_dict = {path: preds for path, preds in zip(image_paths, predictions_list)}
        
        # Calculate statistics
        total_predictions = sum(len(preds) for preds in predictions_list)
        images_with_tags = sum(1 for preds in predictions_list if len(preds) > 0)
        
        print(f"Batch prediction complete: {images_with_tags}/{len(image_paths)} images tagged with {total_predictions} total predictions")
        if progress_callback:
            progress_callback('detail', f'✓ Tagged {images_with_tags}/{len(image_paths)} images ({total_predictions} predictions)')
        
        return predictions_dict
    
    def apply_predicted_tags(self, image_path: str, predictions: List[Dict[str, Any]]):  
        """Apply predicted tags to image in database and file metadata"""
        if not predictions:
            print(f"⚠ No predictions to apply for {Path(image_path).name}")
            return
        
        # Get or create image record
        image_record = metadata_store.get_image_by_path(image_path)
        if image_record is None:
            path_obj = Path(image_path)
            image_id = metadata_store.add_image(
                file_path=image_path,
                file_name=path_obj.name,
                file_size=path_obj.stat().st_size if path_obj.exists() else 0
            )
        else:
            image_id = image_record['id']
        
        # Add predicted tags to database
        tags_added = []
        if image_id is not None:
            for pred in predictions:
                tag_id = metadata_store.get_or_create_tag(pred['tag_name'])
                metadata_store.add_image_tag(
                    image_id=image_id,
                    tag_id=tag_id,
                    confidence=pred['confidence'],
                    source='model_prediction'
                )
                tags_added.append(f"{pred['tag_name']} ({pred['confidence']:.2f})")
        
            metadata_store.update_last_processed(image_id)
        
        # Write tags to image file metadata
        tag_names = [pred['tag_name'] for pred in predictions]
        self.write_tags_to_file(image_path, tag_names)
        
        file_name = Path(image_path).name
        print(f"✓ Tagged {file_name}: {', '.join(tags_added)}")
    
    def write_tags_to_file(self, image_path: str, tags: List[str], silent: bool = False):
        """Write tags to image file metadata (EXIF/IPTC keywords)"""
        if not tags or not os.path.exists(image_path):
            return
        
        try:
            path_obj = Path(image_path)
            file_ext = path_obj.suffix.lower()
            
            # For JPEG/JFIF files, use EXIF
            if file_ext in ['.jpg', '.jpeg', '.jfif']:
                self._write_tags_to_jpeg(image_path, tags, silent)
            # For PNG files, use PNG text chunks
            elif file_ext == '.png':
                self._write_tags_to_png(image_path, tags, silent)
            else:
                if not silent:
                    print(f"Tag writing not supported for {file_ext} files (yet)")
        
        except Exception as e:
            if not silent:
                print(f"Failed to write tags to {image_path}: {e}")
    
    def _write_tags_to_jpeg(self, image_path: str, tags: List[str], silent: bool = False):
        """Write tags to JPEG file using EXIF"""
        try:
            # Load existing EXIF data
            try:
                exif_dict = piexif.load(image_path)
            except:
                # Create new EXIF dict if none exists
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            
            # Windows Photo Viewer and other apps read from IFD (0th) tag 0x9c9e (XPKeywords)
            # This is a UTF-16LE encoded string
            keywords_str = ";".join(tags)
            keywords_bytes = keywords_str.encode('utf-16le')
            
            if "0th" not in exif_dict:
                exif_dict["0th"] = {}
            
            # XPKeywords tag (0x9c9e = 40094)
            exif_dict["0th"][0x9c9e] = keywords_bytes
            
            # Also write to UserComment (more universally supported)
            if "Exif" not in exif_dict:
                exif_dict["Exif"] = {}
            
            user_comment = f"Keywords: {keywords_str}"
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment.encode('utf-8')
            
            # Dump and save
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
            
            if not silent:
                print(f"✓ Wrote {len(tags)} tags to {Path(image_path).name}")
        
        except Exception as e:
            if not silent:
                print(f"Failed to write JPEG EXIF tags: {e}")
    
    def _write_tags_to_png(self, image_path: str, tags: List[str], silent: bool = False):
        """Write tags to PNG file using PNG text chunks"""
        try:
            from PIL import PngImagePlugin
            
            img = Image.open(image_path)
            
            # PNG supports text chunks
            metadata = PngImagePlugin.PngInfo()
            
            # Add keywords
            keywords_str = ";".join(tags)
            metadata.add_text("Keywords", keywords_str)
            metadata.add_text("Tags", keywords_str)
            
            # Save with metadata
            img.save(image_path, pnginfo=metadata)
            
            if not silent:
                print(f"✓ Wrote {len(tags)} tags to {Path(image_path).name}")
        
        except Exception as e:
            if not silent:
                print(f"Failed to write PNG tags: {e}")
    
    def apply_manual_tag(self, image_path: str, tag_name: str, is_ground_truth: bool = False):
        """Manually add a tag to an image"""
        # Get or create image record
        image_record = metadata_store.get_image_by_path(image_path)
        if image_record is None:
            path_obj = Path(image_path)
            image_id = metadata_store.add_image(
                file_path=image_path,
                file_name=path_obj.name,
                file_size=path_obj.stat().st_size if path_obj.exists() else 0
            )
        else:
            image_id = image_record['id']
        
        # Add tag
        tag_id = metadata_store.get_or_create_tag(tag_name)
        if image_id is not None:
            metadata_store.add_image_tag(
                image_id=image_id,
                tag_id=tag_id,
                confidence=1.0,
                source='manual'
            )
        
            # Mark as ground truth if specified
            if is_ground_truth:
                metadata_store.add_ground_truth(image_id, tag_id)
        
        # Write all tags to file metadata
        all_tags = self.get_image_tags(image_path)
        tag_names = [tag['tag_name'] for tag in all_tags]
        self.write_tags_to_file(image_path, tag_names)
    
    def remove_tag(self, image_path: str, tag_name: str):
        """Remove a tag from an image"""
        image_record = metadata_store.get_image_by_path(image_path)
        if image_record is None:
            print(f"Image not found: {image_path}")
            return
        
        # Find tag
        cursor = metadata_store.connection.cursor()
        cursor.execute('SELECT id FROM tags WHERE tag_name = ?', (tag_name,))
        result = cursor.fetchone()
        
        if result:
            tag_id = result[0]
            metadata_store.remove_image_tag(image_record['id'], tag_id)
            
            # Update file metadata with remaining tags
            remaining_tags = self.get_image_tags(image_path)
            tag_names = [tag['tag_name'] for tag in remaining_tags]
            self.write_tags_to_file(image_path, tag_names)
    
    def get_image_tags(self, image_path: str) -> List[Dict[str, Any]]:
        """Get all tags for an image"""
        image_record = metadata_store.get_image_by_path(image_path)
        if image_record is None:
            print(f"⚠ Image not found in database: {image_path}")
            return []
        
        tags = metadata_store.get_image_tags(image_record['id'])
        print(f"✓ Found {len(tags)} tags for image ID {image_record['id']}: {image_path}")
        return tags
    
    def get_all_available_tags(self) -> List[str]:
        """Get all available tags in the system"""
        tags = metadata_store.get_all_tags()
        return [tag['tag_name'] for tag in tags]

    # ------------------------------------------------------------------
    # Folder-path based tag extraction
    # ------------------------------------------------------------------

    def get_folder_tags_for_image(self, image_path: str, root_folder: str) -> List[str]:
        """Derive tags from the folder hierarchy between *root_folder* and *image_path*.

        Each folder name between the root and the file is split on whitespace /
        underscores / hyphens so that multi-word folder names produce multiple tags.

        Example:
            root  = /photos
            image = /photos/ANIME/BOKU NO HERO ACADEMIA/img.jpg
            tags  = ['ANIME', 'BOKU', 'NO', 'HERO', 'ACADEMIA']
        """
        tags: List[str] = []
        try:
            image_p = Path(image_path).resolve()
            root_p  = Path(root_folder).resolve()
            rel = image_p.relative_to(root_p)
            # rel.parts[-1] is the filename — skip it
            for folder_name in rel.parts[:-1]:
                # Split on spaces, underscores, hyphens — keep non-empty tokens
                words = [w.strip() for w in re.split(r'[\s_\-]+', folder_name) if w.strip()]
                tags.extend(words)
        except ValueError:
            # image not under root_folder — return empty
            pass
        return tags

    def apply_folder_tags_to_image(self, image_path: str, root_folder: str | None = None) -> List[str]:
        """Apply folder-derived tags to *image_path* in the database.

        Returns the list of tag names that were applied.
        """
        if root_folder is None:
            root_folder = config_manager.app_config.tag_root_folder
            if not root_folder:
                root_folder = config_manager.app_config.default_image_folder

        tags = self.get_folder_tags_for_image(image_path, root_folder)
        for tag in tags:
            self.apply_manual_tag(image_path, tag, is_ground_truth=False)

        if tags:
            print(f"✓ Folder tags applied to {Path(image_path).name}: {tags}")
        else:
            print(f"⚠ No folder tags derived for {Path(image_path).name} (not under root {root_folder}?)")
        return tags


class TrainingManager:
    """Manages training queue and dispatches training tasks"""
    
    def __init__(self):
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False
        self.training_loop = TrainingLoop()
        self.current_task = None
        self.on_training_complete_callback = None
        self.progress_callback: Callable[[str, Any], None] | None = None
    
    def queue_training_task(self, task_type: str, **kwargs):
        """Add a training task to the queue"""
        task = {
            'type': task_type,
            'params': kwargs,
            'queued_at': time.time()
        }
        self.training_queue.put(task)
        print(f"Training task queued: {task_type}")
        
        # Start processing if not already running
        if not self.is_training:
            self.start_processing()
    
    def start_processing(self):
        """Start processing training queue"""
        if self.training_thread is not None and self.training_thread.is_alive():
            return
        
        self.training_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.training_thread.start()
    
    def _process_queue(self):
        """Process training tasks from queue"""
        while not self.training_queue.empty():
            self.is_training = True
            task = self.training_queue.get()
            self.current_task = task
            
            print(f"Processing training task: {task['type']}")
            
            try:
                if task['type'] == 'train_from_database':
                    self._train_from_database(task['params'])
                elif task['type'] == 'train_from_folders':
                    self._train_from_folders(task['params'])
                elif task['type'] == 'fine_tune':
                    self._fine_tune(task['params'])
                elif task['type'] == 'fine_tune_inbox':
                    self._fine_tune_inbox(task['params'])
                elif task['type'] == 'unsupervised_clustering':
                    self._unsupervised_clustering(task['params'])
            except Exception as e:
                import traceback
                print(f"Error processing training task: {e}")
                print(traceback.format_exc())
                if self.progress_callback:
                    self.progress_callback('error', str(e))
            
            self.training_queue.task_done()
        
        self.is_training = False
        self.current_task = None
        
        # Notify completion
        if self.on_training_complete_callback:
            self.on_training_complete_callback()
    
    def _train_from_database(self, params: Dict[str, Any]):
        """Train model using tags from database"""
        print("Training from database tags...")
        if self.progress_callback:
            self.progress_callback('status', 'Loading training data from database...')
        
        image_paths, labels = self.training_loop.prepare_data_from_database(
            progress_callback=self.progress_callback)
        
        if len(image_paths) > 0:
            if self.progress_callback:
                metrics = self.training_loop.train(image_paths, labels, progress_callback=self.progress_callback)
            else:
                metrics = self.training_loop.train(image_paths, labels)
            val_accs = metrics.get('val_accuracies', metrics.get('accuracies', [0]))
            print(f"Training completed. Final validation accuracy: {val_accs[-1]:.2f}%")
            
            # Auto-load the newly trained model
            print("Loading newly trained model...")
            if self.progress_callback:
                self.progress_callback('status', 'Loading trained model...')
            self._auto_load_model()
        else:
            if self.progress_callback:
                self.progress_callback('error', 'No training data available in database')
            print("No training data available in database")
    
    def _train_from_folders(self, params: Dict[str, Any]):
        """Bootstrap training from folder structure"""
        root_folder = params.get('root_folder', config_manager.app_config.default_image_folder)
        print(f"Bootstrapping training from folder structure: {root_folder}")
        
        if self.progress_callback:
            self.progress_callback('status', 'Scanning folders for training data...')
            self.progress_callback('detail', f'Root folder: {root_folder}')
        
        image_paths, labels = self.training_loop.prepare_data_from_folders(
            root_folder, progress_callback=self.progress_callback)
        
        if len(image_paths) > 0:
            if self.progress_callback:
                metrics = self.training_loop.train(image_paths, labels, progress_callback=self.progress_callback)
            else:
                metrics = self.training_loop.train(image_paths, labels)
            # Get final validation accuracy
            val_accs = metrics.get('val_accuracies', [])
            if val_accs:
                print(f"Training completed. Final validation accuracy: {val_accs[-1]:.2f}%")
            else:
                print("Training completed.")
            
            # Auto-load the newly trained model
            print("Loading newly trained model...")
            if self.progress_callback:
                self.progress_callback('status', 'Loading trained model...')
            self._auto_load_model()
        else:
            if self.progress_callback:
                self.progress_callback('error', 'No images found in folder structure')
            print("No images found in folder structure")
    
    def _fine_tune(self, params: Dict[str, Any]):
        """Fine-tune existing model with corrections"""
        print("Fine-tuning model with user corrections...")
        # Similar to train_from_database but loads existing model first
        self._train_from_database(params)

    def _fine_tune_inbox(self, params: Dict[str, Any]):
        """Fine-tune model on a small set of user-labeled inbox images."""
        annotations: Dict[str, List[str]] = params.get('annotations', {})
        epochs: int = int(params.get('epochs', 5))

        if not annotations:
            print('No annotations provided for inbox fine-tuning')
            return

        print(f'Fine-tuning model on {len(annotations)} inbox image(s), {epochs} epoch(s)…')

        # Use print-only mode (no UI callback) for inline fine-tune so that a
        # stale TrainingProgressDialog callback from a previous training session
        # never tries to update already-destroyed Tkinter widgets.
        try:
            metrics = self.training_loop.fine_tune_from_annotations(
                annotations=annotations,
                epochs=epochs,
                progress_callback=None,
            )
            if metrics:
                losses = metrics.get('losses', [])
                last_loss = losses[-1] if losses else float('nan')
                print(f'Fine-tuning complete — final loss: {last_loss:.4f}')
                self._auto_load_model()
        except Exception as exc:
            import traceback
            print(f'Fine-tuning error: {exc}')
            print(traceback.format_exc())

    def _unsupervised_clustering(self, params: Dict[str, Any]):
        """Perform unsupervised clustering"""
        root_folder = params.get('root_folder', config_manager.app_config.default_image_folder)
        n_clusters = params.get('n_clusters', 10)
        
        print(f"Performing unsupervised clustering with {n_clusters} clusters...")
        
        # Get images
        file_watcher = FileWatcher()
        image_paths = file_watcher.scan_directory(root_folder, recursive=True)
        
        if len(image_paths) > 0:
            clustering = UnsupervisedClustering(n_clusters=n_clusters)
            clusters = clustering.cluster_images(image_paths)
            
            # Apply cluster labels as tags
            for cluster_id, images in clusters.items():
                tag_name = f"cluster_{cluster_id}"
                tag_id = metadata_store.get_or_create_tag(tag_name, category='unsupervised')
                
                for img_path in images:
                    image_record = metadata_store.get_image_by_path(img_path)
                    if image_record:
                        metadata_store.add_image_tag(
                            image_id=image_record['id'],
                            tag_id=tag_id,
                            confidence=1.0,
                            source='clustering'
                        )
            
            print(f"Clustering completed. Created {len(clusters)} clusters")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'queue_size': self.training_queue.qsize(),
            'current_task': self.current_task,
            'metrics': self.training_loop.training_metrics
        }
    
    def _auto_load_model(self):
        """Auto-load the latest trained model into the tagging engine"""
        try:
            # Access the parent logic controller's tagging engine
            from logic_controller import logic_controller
            success = logic_controller.tagging_engine.load_model('latest')
            if success:
                print("✓ Model loaded successfully and ready for inference")
                if self.progress_callback:
                    self.progress_callback('detail', '✓ Model loaded and ready for inference')
            else:
                print("⚠ Failed to load model after training")
                if self.progress_callback:
                    self.progress_callback('detail', '⚠ Model saved but failed to load')
        except Exception as e:
            print(f"⚠ Error loading model after training: {e}")
            if self.progress_callback:
                self.progress_callback('detail', f'⚠ Error loading model: {e}')
    
    def cancel_training(self):
        """Cancel current training (not fully implemented)"""
        # This would require more sophisticated thread management
        print("Training cancellation requested (not fully implemented)")


class LogicController:
    """Main controller that coordinates all logic components"""
    
    def __init__(self):
        self.file_watcher = FileWatcher()
        self.tagging_engine = TaggingEngine()
        self.training_manager = TrainingManager()
        self.current_directory = None
        self.current_images = []
    
    def initialize(self, image_folder: str | None = None):
        """Initialize the controller with an image folder"""
        if image_folder is None:
            image_folder = config_manager.app_config.default_image_folder
        
        self.current_directory = image_folder
        
        # Don't scan automatically - only when user trains/sorts
        self.current_images = []
        
        print(f"Controller initialized with directory: {image_folder}")
        
        # Try to auto-load the latest model if it exists
        self.try_auto_load_model()
    
    def change_directory(self, new_folder: str):
        """Change to a new directory (lightweight - doesn't scan)"""
        print(f"Changing directory to: {new_folder}")
        self.current_directory = new_folder
        
        # Don't scan automatically - only when user trains/sorts
        self.current_images = []
        
        print(f"✓ Directory changed to: {new_folder}")
        print(f"Model remains loaded: {self.tagging_engine.model_loaded}")

    def set_root_folder(self, root_folder: str) -> None:
        """Set the root folder for the session.

        This is the top-level folder the user opens.  The shared database is
        placed here and folder-based tag extraction is relative to this path.
        All sub-folder navigation via :meth:`change_directory` will NOT move
        the database.
        """
        root_folder = str(Path(root_folder).resolve())
        self.current_directory = root_folder

        # Update config
        config_manager.app_config.default_image_folder = root_folder
        config_manager.app_config.db_root_folder       = root_folder
        if not config_manager.app_config.tag_root_folder:
            config_manager.app_config.tag_root_folder = root_folder
        config_manager.save_config()

        # Move the database to the root folder
        metadata_store.set_root_folder(root_folder)
        # Point model + graph storage at <root>/ML_Training_Data/
        model_storage.set_root_folder(root_folder)
        print(f"✓ Root folder set to: {root_folder}")
    
    def start_initial_sorting(self, model_name: str = 'latest', auto_apply: bool = True, progress_callback: Callable | None = None):
        """Start initial sorting operation with progress feedback"""
        # Scan directory for images first
        if progress_callback:
            progress_callback('status', 'Scanning directory for images...')
        
        if self.current_directory:
            self.current_images = self.file_watcher.scan_directory(self.current_directory, recursive=True)
        else:
            self.current_images = []
        self.file_watcher.register_images_to_database(self.current_images)
        
        if progress_callback:
            progress_callback('detail', f'Found {len(self.current_images)} images in directory')
        
        print(f"Found {len(self.current_images)} images for sorting")
        
        if len(self.current_images) == 0:
            error_msg = "No images found in current directory"
            print(f"⚠ {error_msg}")
            if progress_callback:
                progress_callback('error', error_msg)
            return []
        
        # Try to load model
        if progress_callback:
            progress_callback('status', f'Loading model: {model_name}...')
        
        if not self.tagging_engine.load_model(model_name):
            error_msg = "No trained model found. Please train a model first."
            print(f"⚠ {error_msg}")
            if progress_callback:
                progress_callback('error', error_msg)
            return []
        
        if progress_callback:
            progress_callback('status', f'Model loaded successfully: {model_name}')
            progress_callback('detail', f'Starting inference on {len(self.current_images)} images...')
        
        # Predict tags for all images
        print(f"Starting inference on {len(self.current_images)} images...")
        predictions_dict = self.tagging_engine.predict_tags_batch(self.current_images, progress_callback)
        
        # Auto-apply tags if requested
        if auto_apply:
            if progress_callback:
                progress_callback('status', 'Applying predicted tags...')
            
            tagged_count = 0
            for idx, (img_path, predictions) in enumerate(predictions_dict.items()):
                if len(predictions) > 0:
                    self.tagging_engine.apply_predicted_tags(img_path, predictions)
                    tagged_count += 1
                
                # Progress update every 10 images
                if progress_callback and (idx + 1) % 10 == 0:
                    progress = ((idx + 1) / len(predictions_dict)) * 100
                    progress_callback('progress', progress)
                    progress_callback('detail', f'Progress: {idx + 1}/{len(predictions_dict)} images processed')
            
            if progress_callback:
                progress_callback('progress', 100)
                progress_callback('detail', f'✓ Successfully tagged {tagged_count}/{len(self.current_images)} images')
                progress_callback('complete', True)
            
            print(f"✓ Auto-sorting complete: {tagged_count}/{len(self.current_images)} images tagged")
        
        return predictions_dict
    
    def bootstrap_from_folders(self):
        """Bootstrap model from existing folder structure"""
        self.training_manager.queue_training_task(
            'train_from_folders',
            root_folder=self.current_directory
        )
    
    def train_from_corrections(self):
        """Train model from user corrections in database"""
        self.training_manager.queue_training_task('train_from_database')
    
    def perform_unsupervised_clustering(self, n_clusters: int = 10):
        """Perform unsupervised clustering"""
        self.training_manager.queue_training_task(
            'unsupervised_clustering',
            root_folder=self.current_directory,
            n_clusters=n_clusters
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'current_directory': self.current_directory,
            'total_images': len(self.current_images),
            'model_loaded': self.tagging_engine.model_loaded,
            'training_status': self.training_manager.get_training_status()
        }
    
    def try_auto_load_model(self):
        """Try to auto-load the latest model if it exists"""
        try:
            # Check if a model file exists using model_storage
            if model_storage.model_exists('latest'):
                print("Found existing model, loading...")
                success = self.tagging_engine.load_model('latest')
                if success:
                    print("✓ Model auto-loaded successfully")
                else:
                    print("⚠ Model file exists but failed to load")
            else:
                print("No existing model found. Train a model to enable auto-tagging.")
        except Exception as e:
            print(f"⚠ Error during auto-load attempt: {e}")
    
    def get_image_tags(self, image_path: str) -> List[Dict[str, Any]]:
        """Get all tags for an image (delegates to tagging engine)"""
        return self.tagging_engine.get_image_tags(image_path)
    
    def get_all_available_tags(self) -> List[str]:
        """Get all available tags in the system (delegates to tagging engine)"""
        return self.tagging_engine.get_all_available_tags()

    def apply_folder_tags(self, image_path: str, root_folder: str | None = None) -> List[str]:
        """Apply folder-hierarchy tags to an image (delegates to tagging engine)."""
        return self.tagging_engine.apply_folder_tags_to_image(image_path, root_folder)

    # ------------------------------------------------------------------
    # Inbox fine-tuning
    # ------------------------------------------------------------------

    def fine_tune_from_inbox_labels(
        self,
        annotations: Dict[str, List[str]],
        epochs: int = 5,
    ) -> None:
        """Queue a fine-tune task using user-labeled inbox images.

        Args:
            annotations: ``{absolute_image_path: [tag_name, ...]}``
            epochs:       Number of fine-tune passes (default 5).
        """
        self.training_manager.queue_training_task(
            'fine_tune_inbox',
            annotations=annotations,
            epochs=epochs,
        )

    def get_known_tags(self) -> List[str]:
        """Return the list of tag names the current model knows.

        Returns an empty list if no model has been loaded yet.
        """
        try:
            eng = self.tagging_engine.inference_engine
            if eng.label_encoder is not None and len(eng.label_encoder.classes_) > 0:
                return sorted(eng.label_encoder.classes_.tolist())
        except Exception:
            pass
        # Fall back to tags in the database
        return sorted(self.tagging_engine.get_all_available_tags())


# Global controller instance
logic_controller = LogicController()
