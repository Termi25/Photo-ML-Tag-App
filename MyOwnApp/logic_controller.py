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
    
    def __init__(self):
        self.observer = None
        self.watched_paths = set()
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif', '.tiff'}
        self.on_new_images_callback = None
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[str]:
        image_files = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"Directory does not exist: {directory}")
            return []
        
        if recursive:
            for file_path in dir_path.rglob('*'):
                if not (file_path.is_file() and file_path.suffix.lower() in self.image_extensions):
                    continue
                if any(part.lower() == 'thumbs' for part in file_path.relative_to(dir_path).parts[:-1]):
                    continue
                image_files.append(str(file_path))
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                    image_files.append(str(file_path))
        
        print(f"Found {len(image_files)} images in {directory}")
        return image_files
    
    def register_images_to_database(self, image_paths: List[str]):
        if not image_paths:
            return
        try:
            metadata_store.connection.execute("BEGIN")
            for img_path in image_paths:
                path_obj = Path(img_path)
                file_size = path_obj.stat().st_size if path_obj.exists() else 0
                metadata_store.add_image(
                    file_path=str(img_path),
                    file_name=path_obj.name,
                    file_size=file_size,
                    commit=False,
                )
            metadata_store.connection.commit()
        except Exception:
            metadata_store.connection.rollback()
            raise

        print(f"Registered {len(image_paths)} images to database")
    
    def watch_directory(self, directory: str, callback: Callable | None = None):
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
        if self.observer is not None:
            self.observer.stop()
            # Avoid indefinite blocking if the observer thread is stuck.
            try:
                self.observer.join(timeout=2)
            except TypeError:
                # Older watchdog versions may not support a timeout keyword.
                # Use a Thread-level join with timeout as a hard safety net.
                import threading as _threading
                t = _threading.Thread(target=self.observer.join)
                t.daemon = True
                t.start()
                t.join(timeout=2)
            try:
                if hasattr(self.observer, 'is_alive') and self.observer.is_alive():
                    print("⚠ Watcher did not stop within timeout")
            except Exception:
                pass
            self.observer = None
            self.watched_paths.clear()
            print("Stopped watching directories")


class ImageFileEventHandler(FileSystemEventHandler):
    
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
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        self.model_loaded = False
    
    def load_model(self, model_name: str = 'latest') -> bool:
        self.model_loaded = self.inference_engine.load_model(model_name)
        return self.model_loaded
    
    def predict_tags_for_image(self, image_path: str) -> List[Dict[str, Any]]:
        if not self.model_loaded:
            print("No model loaded for inference")
            return []
        
        predictions = self.inference_engine.predict_single(image_path)
        return predictions

    def get_image_tags(self, image_path: str) -> List[Dict[str, Any]]:
        """Return stored tags for an image by querying the metadata store."""
        try:
            rec = metadata_store.get_image_by_path(image_path)
            if rec is None:
                return []
            return metadata_store.get_image_tags(rec['id'])
        except Exception:
            return []

    def get_all_available_tags(self) -> List[str]:
        """Return a list of tag names available in the DB."""
        try:
            return [t['tag_name'] for t in metadata_store.get_all_tags()]
        except Exception:
            return []

    def apply_folder_tags_to_image(self, image_path: str, root_folder: str | None = None) -> List[str]:
        """(Minimal) derive tags from containing folder names. Returns applied tags list."""
        try:
            p = Path(image_path)
            parents = list(p.parents)
            applied = []
            for parent in parents:
                if root_folder and str(parent).startswith(str(Path(root_folder).resolve())):
                    applied.append(parent.name)
            return applied
        except Exception:
            return []
    
    def predict_tags_batch(self, image_paths: List[str], progress_callback: Callable | None = None) -> Dict[str, List[Dict[str, Any]]]:
        if not self.model_loaded:
            print("No model loaded for inference")
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
            progress_callback('detail', f'Tagged {images_with_tags}/{len(image_paths)} images ({total_predictions} predictions)')
        
        return predictions_dict
    
    def apply_predicted_tags(self, image_path: str, predictions: List[Dict[str, Any]]):  
        if not predictions:
            print(f"No predictions to apply for {Path(image_path).name}")
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
        print(f"Tagged {file_name}: {', '.join(tags_added)}")
    
    def write_tags_to_file(self, image_path: str, tags: List[str], silent: bool = False):
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
        try:
            # Load existing EXIF data
            try:
                exif_dict = piexif.load(image_path)
            except Exception:
                # Create new EXIF dict if none exists or data is unreadable
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
                print(f"Wrote {len(tags)} tags to {Path(image_path).name}")
        
        except Exception as e:
            if not silent:
                print(f"Failed to write PNG tags: {e}")
    
    def apply_manual_tag(self, image_path: str, tag_name: str, is_ground_truth: bool = False):
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
    



class TrainingManager:
    def __init__(self):
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False
        self._state_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self.training_loop = TrainingLoop()
        self.current_task = None
        self.on_training_complete_callback = None
        self.progress_callback: Callable[[str, Any], None] | None = None

    def cancel(self) -> None:
        self._cancel_event.set()

    def _emit(self, kind: str, value) -> None:
        cb = self.progress_callback
        if cb:
            cb(kind, value)

    def queue_training_task(self, task_type: str, **kwargs):
        task = {'type': task_type, 'params': kwargs, 'queued_at': time.time()}
        self.training_queue.put(task)
        print(f"Training task queued: {task_type}")
        with self._state_lock:
            should_start = not self.is_training and not (
                self.training_thread is not None and self.training_thread.is_alive()
            )
        if should_start:
            self.start_processing()

    def start_processing(self):
        with self._state_lock:
            if self.training_thread is not None and self.training_thread.is_alive():
                return
            self.is_training = True
            self.training_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.training_thread.start()

    def _process_queue(self):
        self._cancel_event.clear()
        while True:
            try:
                task = self.training_queue.get_nowait()
            except queue.Empty:
                break
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
                elif task['type'] == 'train_from_coco':
                    self._train_from_coco(task['params'])
                elif task['type'] == 'train_from_both':
                    self._train_from_both(task['params'])
            except Exception as e:
                import traceback
                print(f"Error processing training task: {e}")
                print(traceback.format_exc())
                if self.progress_callback:
                    self.progress_callback('error', str(e))
            self.training_queue.task_done()

        with self._state_lock:
            self.is_training = False
        self.current_task = None
        if self.on_training_complete_callback:
            self.on_training_complete_callback()

    def _train_from_database(self, params: Dict[str, Any]):
        _cb = self.progress_callback
        self.training_loop.data_source = 'database'
        self._emit('status', 'Loading training data from database...')

        image_paths, labels = self.training_loop.prepare_data_from_database(
            progress_callback=_cb)

        if not image_paths:
            self._emit('error', 'No training data available in database')
            return

        metrics = self.training_loop.train(
            image_paths, labels,
            test_split=0.15,
            progress_callback=_cb,
            cancel_event=self._cancel_event,
        )
        if metrics.get('cancelled'):
            return

        val_accs = metrics.get('val_accuracies') or metrics.get('accuracies') or []
        if val_accs:
            print(f"Training completed. Final val accuracy: {val_accs[-1]:.2f}%")
        self._report_test_results(metrics)
        self._emit('status', 'Loading trained model...')
        self._auto_load_model()
        self._emit('complete', True)

    def _train_from_folders(self, params: Dict[str, Any]):
        _cb = self.progress_callback
        self.training_loop.data_source = 'folders'
        root_folder = params.get('root_folder', config_manager.app_config.default_image_folder)
        self._emit('status', 'Scanning folders for training data...')
        self._emit('detail', f'Root folder: {root_folder}')

        image_paths, labels = self.training_loop.prepare_data_from_folders(
            root_folder, progress_callback=_cb)

        if not image_paths:
            self._emit('error', 'No images found in folder structure')
            return

        metrics = self.training_loop.train(
            image_paths, labels,
            test_split=0.15,
            progress_callback=_cb,
            cancel_event=self._cancel_event,
        )
        if metrics.get('cancelled'):
            return

        val_accs = metrics.get('val_accuracies', [])
        if val_accs:
            print(f"Training completed. Final val accuracy: {val_accs[-1]:.2f}%")
        self._report_test_results(metrics)
        self._emit('status', 'Loading trained model...')
        self._auto_load_model()
        self._emit('complete', True)

    def _train_from_coco(self, params: Dict[str, Any]):
        _cb = self.progress_callback
        self.training_loop.data_source = 'coco'
        coco_dir = params.get('coco_dir', config_manager.app_config.coco_data_dir or './data/coco')
        coco_max = int(params.get('coco_max', 0))
        self._emit('status', 'Loading COCO dataset...')
        self._emit('detail', f'COCO directory: {coco_dir}')
        self._emit('detail', f'Max images: {coco_max if coco_max > 0 else 5000}')

        from coco_benchmark import COCOBenchmarkLoader
        max_images = coco_max if coco_max > 0 else 5000
        loader = COCOBenchmarkLoader(coco_dir, max_images=max_images)
        image_paths, labels = loader.load_data(progress_callback=_cb)

        if not image_paths:
            self._emit('error', 'No COCO images could be loaded')
            return

        metrics = self.training_loop.train(
            image_paths, labels,
            test_split=0.15,
            progress_callback=_cb,
            cancel_event=self._cancel_event,
        )
        if metrics.get('cancelled'):
            return

        self._report_test_results(metrics)
        self._emit('status', 'Loading trained model...')
        self._auto_load_model()
        self._emit('complete', True)

    def _train_from_both(self, params: Dict[str, Any]):
        _cb = self.progress_callback
        self.training_loop.data_source = 'both'
        root_folder = params.get('root_folder', config_manager.app_config.default_image_folder)
        coco_dir = params.get('coco_dir', config_manager.app_config.coco_data_dir or './data/coco')
        coco_max = int(params.get('coco_max', 0))

        self._emit('status', 'Loading folder data...')
        folder_paths, folder_labels = self.training_loop.prepare_data_from_folders(
            root_folder, progress_callback=_cb)

        self._emit('status', 'Loading COCO dataset...')
        from coco_benchmark import COCOBenchmarkLoader
        max_images = coco_max if coco_max > 0 else 5000
        loader = COCOBenchmarkLoader(coco_dir, max_images=max_images)
        coco_paths, coco_labels = loader.load_data(progress_callback=_cb)

        image_paths = folder_paths + coco_paths
        labels = folder_labels + coco_labels

        if not image_paths:
            self._emit('error', 'No training data found from folders or COCO')
            return

        metrics = self.training_loop.train(
            image_paths, labels,
            test_split=0.15,
            progress_callback=_cb,
            cancel_event=self._cancel_event,
        )
        if metrics.get('cancelled'):
            return

        self._report_test_results(metrics)
        self._emit('status', 'Loading trained model...')
        self._auto_load_model()
        self._emit('complete', True)

    def _fine_tune(self, params: Dict[str, Any]):
        self._train_from_database(params)

    def _fine_tune_inbox(self, params: Dict[str, Any]):
        _cb = self.progress_callback
        self.training_loop.data_source = 'inbox'
        annotations: Dict[str, List[str]] = params.get('annotations', {})
        epochs: int = int(params.get('epochs', 5))
        if not annotations:
            return
        try:
            metrics = self.training_loop.fine_tune_from_annotations(
                annotations=annotations, epochs=epochs,
                progress_callback=_cb)
            if metrics:
                losses = metrics.get('losses', [])
                print(f'Fine-tuning complete — final loss: {losses[-1]:.4f}' if losses else 'Fine-tuning complete')
                self._auto_load_model()
            self._emit('complete', True)
        except Exception as exc:
            import traceback
            print(f'Fine-tuning error: {exc}')
            print(traceback.format_exc())
            self._emit('error', str(exc))

    def _unsupervised_clustering(self, params: Dict[str, Any]):
        root_folder = params.get('root_folder', config_manager.app_config.default_image_folder)
        n_clusters = params.get('n_clusters', 10)
        file_watcher = FileWatcher()
        image_paths = file_watcher.scan_directory(root_folder, recursive=True)
        if len(image_paths) > 0:
            clustering = UnsupervisedClustering(n_clusters=n_clusters)
            clusters = clustering.cluster_images(image_paths)
            for cluster_id, images in clusters.items():
                tag_name = f"cluster_{cluster_id}"
                tag_id = metadata_store.get_or_create_tag(tag_name, category='unsupervised')
                for img_path in images:
                    image_record = metadata_store.get_image_by_path(img_path)
                    if image_record:
                        metadata_store.add_image_tag(
                            image_id=image_record['id'], tag_id=tag_id,
                            confidence=1.0, source='clustering')
            print(f"Clustering completed. Created {len(clusters)} clusters")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'queue_size': self.training_queue.qsize(),
            'current_task': self.current_task,
            'metrics': self.training_loop.training_metrics
        }
    
    def _report_test_results(self, metrics: Dict[str, Any]) -> None:
        """Push held-out test-set results to the progress dialog (if available)."""
        test = metrics.get('test_results')
        if not test or 'error' in test:
            return
        _cb = self.progress_callback
        if not _cb:
            return
        _cb('status', 'Test-set evaluation complete')
        _cb('detail', '── Held-out Test Set Results ──────────────────')
        _cb('detail', f'  Images evaluated : {test.get("n_test_images", "?")}')
        _cb('detail', f'  Accuracy         : {test.get("test_accuracy", 0):.2f}%')
        _cb('detail', f'  Macro F1         : {test.get("macro_f1", 0):.4f}')
        _cb('detail', f'  Micro F1         : {test.get("micro_f1", 0):.4f}')
        per_class = test.get('per_class_f1', {})
        if per_class:
            top = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)[:5]
            bottom = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
            _cb('detail', '  Top-5 classes by F1:')
            for cls, sc in top:
                _cb('detail', f'    {cls}: {sc:.4f}')
            _cb('detail', '  Bottom-5 classes by F1:')
            for cls, sc in bottom:
                _cb('detail', f'    {cls}: {sc:.4f}')
        _cb('detail', '───────────────────────────────────────────────')

    def _auto_load_model(self):
        if not config_manager.app_config.auto_load_model:
            return
        try:
            # Avoid importing this module from itself (can stall under the
            # import lock in rare cases); use the already-created global.
            controller = globals().get('logic_controller')
            if controller is None:
                return
            success = controller.tagging_engine.load_model('latest')
            msg = '✓ Model loaded and ready for inference' if success else '⚠ Model saved but failed to load'
            self._emit('detail', msg)
        except Exception as e:
            self._emit('detail', f'⚠ Error loading model: {e}')

    def cancel_training(self):
        self.cancel()


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
        
        # Do not auto-load model on initialize; make loading user-triggered.
        # Call `try_auto_load_model()` manually when needed (e.g., via UI).
    
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
    
    def bootstrap_from_coco(self, coco_dir: str = '', coco_max: int = 0):
        self.training_manager.queue_training_task(
            'train_from_coco',
            coco_dir=coco_dir or config_manager.app_config.coco_data_dir or './data/coco',
            coco_max=coco_max,
        )

    def bootstrap_from_both(self, coco_dir: str = '', coco_max: int = 0):
        self.training_manager.queue_training_task(
            'train_from_both',
            root_folder=self.current_directory,
            coco_dir=coco_dir or config_manager.app_config.coco_data_dir or './data/coco',
            coco_max=coco_max,
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
        if not config_manager.app_config.auto_load_model:
            print("Auto-load disabled by configuration; skipping auto-load attempt.")
            return False
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

    def predict_tags_for_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Delegate single-image prediction to the tagging engine."""
        return self.tagging_engine.predict_tags_for_image(image_path)
    
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
