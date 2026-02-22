"""
Logic Controller
Orchestration layer that coordinates File Watcher, Tagging Engine, and Training Manager.
Acts as the bridge between UI and ML Module.
"""

import os
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

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
    
    def watch_directory(self, directory: str, callback: Callable = None):
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
    
    def __init__(self, image_extensions: set, callback: Callable = None):
        self.image_extensions = image_extensions
        self.callback = callback
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
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
    
    def predict_tags_batch(self, image_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Predict tags for multiple images"""
        if not self.model_loaded:
            print("No model loaded for inference")
            return {path: [] for path in image_paths}
        
        predictions_list = self.inference_engine.predict_batch(image_paths)
        return {path: preds for path, preds in zip(image_paths, predictions_list)}
    
    def apply_predicted_tags(self, image_path: str, predictions: List[Dict[str, Any]]):
        """Apply predicted tags to image in database"""
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
        
        # Add predicted tags
        for pred in predictions:
            tag_id = metadata_store.get_or_create_tag(pred['tag_name'])
            metadata_store.add_image_tag(
                image_id=image_id,
                tag_id=tag_id,
                confidence=pred['confidence'],
                source='model_prediction'
            )
        
        metadata_store.update_last_processed(image_id)
    
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
        metadata_store.add_image_tag(
            image_id=image_id,
            tag_id=tag_id,
            confidence=1.0,
            source='manual'
        )
        
        # Mark as ground truth if specified
        if is_ground_truth:
            metadata_store.add_ground_truth(image_id, tag_id)
    
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
    
    def get_image_tags(self, image_path: str) -> List[Dict[str, Any]]:
        """Get all tags for an image"""
        image_record = metadata_store.get_image_by_path(image_path)
        if image_record is None:
            return []
        
        return metadata_store.get_image_tags(image_record['id'])
    
    def get_all_available_tags(self) -> List[str]:
        """Get all available tags in the system"""
        tags = metadata_store.get_all_tags()
        return [tag['tag_name'] for tag in tags]


class TrainingManager:
    """Manages training queue and dispatches training tasks"""
    
    def __init__(self):
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_training = False
        self.training_loop = TrainingLoop()
        self.current_task = None
        self.on_training_complete_callback = None
        self.progress_callback = None
    
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
                elif task['type'] == 'unsupervised_clustering':
                    self._unsupervised_clustering(task['params'])
            except Exception as e:
                print(f"Error processing training task: {e}")
            
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
        
        image_paths, labels = self.training_loop.prepare_data_from_database()
        
        if len(image_paths) > 0:
            metrics = self.training_loop.train(image_paths, labels, progress_callback=self.progress_callback)
            print(f"Training completed. Final accuracy: {metrics.get('accuracies', [0])[-1]:.2f}%")
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
        
        image_paths, labels = self.training_loop.prepare_data_from_folders(root_folder)
        
        if len(image_paths) > 0:
            metrics = self.training_loop.train(image_paths, labels, progress_callback=self.progress_callback)
            print(f"Training completed. Final accuracy: {metrics.get('accuracies', [0])[-1]:.2f}%")
        else:
            if self.progress_callback:
                self.progress_callback('error', 'No images found in folder structure')
            print("No images found in folder structure")
    
    def _fine_tune(self, params: Dict[str, Any]):
        """Fine-tune existing model with corrections"""
        print("Fine-tuning model with user corrections...")
        # Similar to train_from_database but loads existing model first
        self._train_from_database(params)
    
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
    
    def initialize(self, image_folder: str = None):
        """Initialize the controller with an image folder"""
        if image_folder is None:
            image_folder = config_manager.app_config.default_image_folder
        
        self.current_directory = image_folder
        
        # Scan and register images
        self.current_images = self.file_watcher.scan_directory(image_folder, recursive=True)
        self.file_watcher.register_images_to_database(self.current_images)
        
        print(f"Controller initialized with {len(self.current_images)} images")
    
    def start_initial_sorting(self, model_name: str = 'latest', auto_apply: bool = True):
        """Start initial sorting operation"""
        # Try to load model
        if not self.tagging_engine.load_model(model_name):
            print("No trained model found. Please train a model first.")
            return []
        
        # Predict tags for all images
        print(f"Starting inference on {len(self.current_images)} images...")
        predictions_dict = self.tagging_engine.predict_tags_batch(self.current_images)
        
        # Auto-apply tags if requested
        if auto_apply:
            for img_path, predictions in predictions_dict.items():
                if len(predictions) > 0:
                    self.tagging_engine.apply_predicted_tags(img_path, predictions)
        
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


# Global controller instance
logic_controller = LogicController()
