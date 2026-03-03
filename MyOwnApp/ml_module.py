"""
Machine Learning Module
Handles image preprocessing, inference engine, and training loop for automatic image sorting.
Supports both supervised (tag-based) and unsupervised (clustering) approaches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import time
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from config import config_manager
from data_layer import metadata_store, model_storage

# Optional matplotlib — used to save training graphs
try:
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend, safe in background threads
    import matplotlib.pyplot as _plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _plt = None  # type: ignore[assignment]
    _MATPLOTLIB_AVAILABLE = False

# Increase PIL decompression bomb limit for large images
Image.MAX_IMAGE_PIXELS = 200000000  # Allow up to 200 megapixels

# Suppress Pillow warnings for palette transparency
import warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
warnings.filterwarnings('ignore', message='Palette images with Transparency')


class ImagePreprocessor:
    """Handles image preprocessing and normalization"""
    
    def __init__(self):
        cfg = config_manager.hyperparameters
        self.image_size = cfg.image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.normalization_mean, std=cfg.normalization_std)
        ])
        
        # For inference with timing
        self.preprocessing_times = []
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:  # type: ignore
        """Preprocess a single image"""
        start_time = time.time()
        
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img)
            
            elapsed = time.time() - start_time
            self.preprocessing_times.append(elapsed)
            
            return tensor  # type: ignore
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None  # type: ignore
    
    def preprocess_batch(self, image_paths: List[str]) -> torch.Tensor | None:
        """Preprocess a batch of images"""
        tensors = []
        for path in image_paths:
            tensor = self.preprocess_image(path)
            if tensor is not None:
                tensors.append(tensor)
        
        if len(tensors) == 0:
            return None
        
        return torch.stack(tensors)
    
    def get_avg_preprocessing_time(self) -> float:
        """Get average preprocessing time"""
        if len(self.preprocessing_times) == 0:
            return 0.0
        return sum(self.preprocessing_times) / len(self.preprocessing_times)


class ImageDataset(Dataset):
    """Custom dataset for image classification (supports both single-label and multi-label)"""
    
    def __init__(self, image_paths: List[str], labels: List[int] | List[List[float]] | None = None, transform=None, multi_label: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.multi_label = multi_label
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a blank tensor for missing/corrupt images so the DataLoader never hangs
            cfg = config_manager.hyperparameters
            image = Image.new('RGB', cfg.image_size, (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            if self.multi_label:
                # For multi-label, labels are already binary vectors
                return image, torch.FloatTensor(self.labels[idx])
            else:
                # For single-label, labels are class indices
                return image, self.labels[idx]
        return image


class ImageClassificationModel(nn.Module):
    """Neural network model for image classification with support for multiple architectures"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()
        cfg = config_manager.hyperparameters
        model_type = config_manager.app_config.model_type
        
        # Load backbone based on model type
        if model_type == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore
            
        elif model_type == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore
            
        elif model_type == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore
            
        elif model_type == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore
            
        elif model_type == 'efficientnet_b1':
            from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b1(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore
            
        elif model_type == 'mobilenet_v2':
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = mobilenet_v2(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore
            
        elif model_type == 'vgg16':
            from torchvision.models import vgg16, VGG16_Weights
            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = vgg16(weights=weights)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier = nn.Identity()  # type: ignore
            
        elif model_type == 'resnet':
            # Legacy support
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore
            
        else:
            # Simple custom CNN for unknown types
            print(f"Warning: Unknown model type '{model_type}', using custom CNN")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            num_features = 128
        
        # Classification head
        layers = []
        in_features = int(num_features)  # type: ignore
        
        hidden_layers = cfg.hidden_layers if cfg.hidden_layers is not None else []
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout_rate))
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """Extract features without classification"""
        return self.backbone(x)


class InferenceEngine:
    """Handles model inference for tag prediction (supports both single-label and multi-label)"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.multi_label_mode = False
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                   config_manager.app_config.use_gpu else 'cpu')
        self.preprocessor = ImagePreprocessor()
        self.inference_times = []
    
    def load_model(self, model_name: str = 'latest'):
        """Load trained model"""
        try:
            model_state, metadata = model_storage.load_model_weights(model_name)
            
            num_classes = metadata.get('num_classes', 10)
            self.multi_label_mode = metadata.get('multi_label', False)
            
            self.model = ImageClassificationModel(num_classes=num_classes)
            self.model.load_state_dict(model_state['model_state'])
            self.model.to(self.device)
            self.model.eval()
            
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(metadata.get('classes', []))
            
            mode_str = 'multi-label' if self.multi_label_mode else 'single-label'
            print(f"Model loaded: {model_name} with {num_classes} classes ({mode_str})")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single(self, image_path: str) -> List[Dict[str, Any]]:
        """Predict tags for a single image"""
        if self.model is None:
            print("No model loaded")
            return []
        
        start_time = time.time()
        
        # Preprocess
        tensor = self.preprocessor.preprocess_image(image_path)
        if tensor is None:
            return []
        
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            
            if self.multi_label_mode:
                # Multi-label: use sigmoid activation
                probabilities = torch.sigmoid(outputs)
            else:
                # Single-label: use softmax activation
                probabilities = torch.softmax(outputs, dim=1)
        
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        
        # Get predictions
        cfg = config_manager.hyperparameters
        predictions = []
        
        if self.multi_label_mode:
            # Multi-label: return all tags above threshold
            probs = probabilities[0].cpu().numpy()
            if self.label_encoder is not None:
                for idx, prob in enumerate(probs):
                    if prob >= cfg.confidence_threshold:
                        tag_name = self.label_encoder.classes_[idx]
                        predictions.append({
                            'tag_name': tag_name,
                            'confidence': float(prob)
                        })
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
        else:
            # Single-label: return top-k predictions
            topk_prob, topk_indices = torch.topk(probabilities[0], 
                                                 min(cfg.max_suggestions, len(probabilities[0])))
            
            if self.label_encoder is not None:
                for prob, idx in zip(topk_prob.cpu().numpy(), topk_indices.cpu().numpy()):
                    if prob >= cfg.confidence_threshold:
                        tag_name = self.label_encoder.classes_[idx]
                        predictions.append({
                            'tag_name': tag_name,
                            'confidence': float(prob)
                        })
        
        return predictions
    
    def predict_batch(self, image_paths: List[str]) -> List[List[Dict[str, Any]]]:
        """Predict tags for multiple images"""
        if self.model is None:
            return [[] for _ in image_paths]
        
        start_time = time.time()
        
        # Preprocess batch
        tensors = self.preprocessor.preprocess_batch(image_paths)
        if tensors is None:
            return [[] for _ in image_paths]
        
        tensors = tensors.to(self.device)
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(tensors)
            
            if self.multi_label_mode:
                # Multi-label: use sigmoid activation
                probabilities = torch.sigmoid(outputs)
            else:
                # Single-label: use softmax activation
                probabilities = torch.softmax(outputs, dim=1)
        
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        
        # Process predictions for each image
        batch_predictions = []
        cfg = config_manager.hyperparameters
        
        for prob_dist in probabilities:
            predictions = []
            
            if self.multi_label_mode:
                # Multi-label: return all tags above threshold
                probs = prob_dist.cpu().numpy()
                if self.label_encoder is not None:
                    for idx, prob in enumerate(probs):
                        if prob >= cfg.confidence_threshold:
                            tag_name = self.label_encoder.classes_[idx]
                            predictions.append({
                                'tag_name': tag_name,
                                'confidence': float(prob)
                            })
                # Sort by confidence
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
            else:
                # Single-label: return top-k predictions
                topk_prob, topk_indices = torch.topk(prob_dist, 
                                                     min(cfg.max_suggestions, len(prob_dist)))
                
                if self.label_encoder is not None:
                    for prob, idx in zip(topk_prob.cpu().numpy(), topk_indices.cpu().numpy()):
                        if prob >= cfg.confidence_threshold:
                            tag_name = self.label_encoder.classes_[idx]
                            predictions.append({
                                'tag_name': tag_name,
                                'confidence': float(prob)
                            })
            
            batch_predictions.append(predictions)
        
        return batch_predictions
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time"""
        if len(self.inference_times) == 0:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)


class TrainingLoop:
    """Handles model training with backpropagation (supports multi-label classification)"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                   config_manager.app_config.use_gpu else 'cpu')
        self.training_metrics = {
            'losses': [],
            'accuracies': [],
            'epochs_completed': 0,
            'total_time': 0.0
        }
        self.multi_label_mode = False  # Track if we're in multi-label mode
    
    def prepare_data_from_database(self,
                                    progress_callback: Callable | None = None
                                    ) -> Tuple[List[str], List[List[float]]]:
        """Prepare multi-label training data from all database tags"""
        if progress_callback:
            progress_callback('status', 'Loading training data from database...')
        
        cursor = metadata_store.connection.cursor()
        cursor.execute('''
            SELECT i.file_path, GROUP_CONCAT(t.tag_name, '|||') as tags
            FROM images i
            JOIN image_tags it ON i.id = it.image_id
            JOIN tags t ON it.tag_id = t.id
            GROUP BY i.id
            HAVING tags IS NOT NULL
        ''')
        rows = cursor.fetchall()
        
        if not rows:
            print("No training data found in database")
            return [], []
        
        # Filter to files that actually exist on disk
        if progress_callback:
            progress_callback('status', f'Verifying {len(rows)} image paths...')
        
        image_paths: List[str] = []
        all_tags_per_image: List[List[str]] = []
        skipped = 0
        
        for row in rows:
            file_path = row[0]
            if not Path(file_path).exists():
                skipped += 1
                continue
            tags = [t.strip() for t in row[1].split('|||') if t.strip()]
            if not tags:
                continue
            image_paths.append(file_path)
            all_tags_per_image.append(tags)
        
        if skipped:
            print(f"Skipped {skipped} missing files from database")
            if progress_callback:
                progress_callback('detail', f'Skipped {skipped} files no longer on disk')
        
        if len(image_paths) == 0:
            print("No training data found in database")
            return [], []
        
        # Build multi-label vectors (same format as folder training)
        all_unique_tags: set = set()
        for tags in all_tags_per_image:
            all_unique_tags.update(tags)
        sorted_tags = sorted(all_unique_tags)
        self.label_encoder.classes_ = np.array(sorted_tags)
        
        tag_to_idx = {tag: idx for idx, tag in enumerate(sorted_tags)}
        multi_labels: List[List[float]] = []
        for tags in all_tags_per_image:
            vec = [0.0] * len(sorted_tags)
            for tag in tags:
                vec[tag_to_idx[tag]] = 1.0
            multi_labels.append(vec)
        
        print(f"Prepared {len(image_paths)} images with {len(sorted_tags)} unique tags (multi-label)")
        if progress_callback:
            progress_callback('detail', f'Images: {len(image_paths)}, Tags: {len(sorted_tags)}')
            progress_callback('progress', 5)  # Small initial bump before training starts
        
        return image_paths, multi_labels
    
    def prepare_data_from_folders(self, root_folder: str,
                                    progress_callback: Callable | None = None) -> Tuple[List[str], List[List[float]]]:
        """Bootstrap training data from folder structure with hierarchical multi-label tags"""
        root_path = Path(root_folder)
        image_paths = []
        all_tags_per_image = []  # List of tag lists for each image
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif'}
        
        # First pass: collect all images and their hierarchical tags
        print("Scanning folders and extracting hierarchical tags...")
        if progress_callback:
            progress_callback('status', 'Scanning folder structure...')
        for img_file in root_path.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                # Skip hidden/system files (e.g. .folder_preview.png)
                if img_file.name.startswith('.'):
                    continue
                
                # Get all parent folders from root to image's parent
                relative_path = img_file.relative_to(root_path)
                parent_folders = list(relative_path.parent.parts)  # All parent folder names
                
                # Skip images inside the _INBOX folder
                if '_INBOX' in parent_folders:
                    continue
                
                if parent_folders and parent_folders[0] != '.':  # Has at least one parent folder
                    image_paths.append(str(img_file))
                    all_tags_per_image.append(parent_folders)
        
        if len(image_paths) == 0:
            print("No images found in folder structure")
            return [], []
        
        # Collect all unique tags and create label encoder
        all_unique_tags = set()
        for tags in all_tags_per_image:
            all_unique_tags.update(tags)
        
        # Sort tags for consistent ordering
        sorted_tags = sorted(list(all_unique_tags))
        self.label_encoder.classes_ = np.array(sorted_tags)
        
        print(f"Found {len(sorted_tags)} unique tags: {', '.join(sorted_tags[:10])}{'...' if len(sorted_tags) > 10 else ''}")
        
        # Create multi-label binary vectors
        multi_labels = []
        tag_to_idx = {tag: idx for idx, tag in enumerate(sorted_tags)}
        
        for tags in all_tags_per_image:
            # Create binary vector for this image
            label_vector = [0.0] * len(sorted_tags)
            for tag in tags:
                label_vector[tag_to_idx[tag]] = 1.0
            multi_labels.append(label_vector)
        
        # Register these tags in the database for supervised learning
        from data_layer import metadata_store
        from logic_controller import TaggingEngine
        
        total_to_register = len(image_paths)
        print(f"Registering {total_to_register} images and their hierarchical tags in database...")
        if progress_callback:
            progress_callback('status', f'Registering {total_to_register} images in database...')
        tagging_engine = TaggingEngine()  # Temporary instance for writing tags
        
        # Wrap all per-image DB writes in a single transaction so we do one
        # fsync instead of N*M, avoiding apparent freezes on large folders.
        try:
            metadata_store.connection.execute("BEGIN")
            for reg_idx, (img_path, tags) in enumerate(zip(image_paths, all_tags_per_image)):
                path_obj = Path(img_path)
                # Add image to database if not exists (commit=False: part of batch)
                image_id = metadata_store.add_image(
                    file_path=str(img_path),
                    file_name=path_obj.name,
                    file_size=path_obj.stat().st_size if path_obj.exists() else 0,
                    commit=False
                )
                # Add all hierarchical tags as ground truth for supervised learning
                if image_id is not None:
                    for tag in tags:
                        tag_id = metadata_store.get_or_create_tag(
                            tag, category='folder_supervised', commit=False)
                        metadata_store.add_image_tag(
                            image_id=image_id,
                            tag_id=tag_id,
                            confidence=1.0,
                            source='folder_supervised',
                            commit=False
                        )
                
                # Write tags to image file metadata so they persist (silent during batch)
                # Wrapped in try/except to avoid hangs on unsupported file formats
                try:
                    tagging_engine.write_tags_to_file(img_path, tags, silent=True)
                except Exception:
                    pass
                
                # Report progress every 10 images
                if progress_callback and (reg_idx % 10 == 0 or reg_idx == total_to_register - 1):
                    pct = int(100 * (reg_idx + 1) / total_to_register)
                    progress_callback('status', f'Registering images: {reg_idx + 1}/{total_to_register} ({pct}%)')
                    progress_callback('progress', pct * 0.3)  # Pre-training phase = 0-30%
            # Single commit for all N images
            metadata_store.connection.commit()
        except Exception:
            metadata_store.connection.rollback()
            raise
        
        print(f"\nHierarchical tagging examples:")
        for i in range(min(3, len(image_paths))):
            file_name = Path(image_paths[i]).name
            tags_str = ', '.join(all_tags_per_image[i])
            print(f"  {file_name} -> [{tags_str}]")
        
        print(f"\nBootstrapped {len(image_paths)} images with {len(sorted_tags)} unique tags (multi-label)")
        return image_paths, multi_labels
    
    def _stratified_split(
        self,
        image_paths: List[str],
        labels: list,
        validation_split: float,
    ) -> Tuple[List[str], List[str], list, list]:
        """Split data into train/val with class-balanced representation.

        * Single-label  → sklearn StratifiedShuffleSplit (exact stratification).
        * Multi-label   → iterative stratification approximation so every tag
          appears in both sets with roughly the same proportion.
        Both modes shuffle with a fixed seed for reproducibility.
        """
        n = len(image_paths)
        
        if self.multi_label_mode:
            label_matrix = np.array(labels, dtype=np.float32)  # (n, n_tags)
            n_tags = label_matrix.shape[1]
            
            # Desired number of positives per tag in the val set
            tag_counts = label_matrix.sum(axis=0)
            desired_val = tag_counts * validation_split
            
            # Shuffle indices with fixed seed so order is random but reproducible
            rng = np.random.default_rng(42)
            indices = rng.permutation(n).tolist()
            
            # Iterative stratification: for each sample pick the split
            # that most needs its rarest active tag.
            val_size = max(1, int(round(n * validation_split)))
            val_tag_counts = np.zeros(n_tags, dtype=np.float32)
            train_idx: List[int] = []
            val_idx: List[int] = []
            
            for idx in indices:
                active = np.where(label_matrix[idx] == 1)[0]
                if len(active) == 0:
                    # No tags: assign proportionally
                    if len(val_idx) < val_size:
                        val_idx.append(idx)
                    else:
                        train_idx.append(idx)
                    continue
                
                # Deficit = how many more positives val still needs for each tag
                deficit = desired_val[active] - val_tag_counts[active]
                if len(val_idx) < val_size and deficit.max() > 0:
                    val_idx.append(idx)
                    val_tag_counts[active] += 1
                else:
                    train_idx.append(idx)
            
            train_paths  = [image_paths[i] for i in train_idx]
            val_paths    = [image_paths[i] for i in val_idx]
            train_labels = [labels[i]      for i in train_idx]
            val_labels   = [labels[i]      for i in val_idx]
        else:
            # Single-label: sklearn stratified split
            from sklearn.model_selection import train_test_split
            try:
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    image_paths, labels,
                    test_size=validation_split,
                    stratify=labels,
                    random_state=42,
                )
            except ValueError:
                # Fallback: too few samples per class to stratify — shuffle + split
                from sklearn.model_selection import train_test_split as tts
                train_paths, val_paths, train_labels, val_labels = tts(
                    image_paths, labels,
                    test_size=validation_split,
                    random_state=42,
                )
        
        return train_paths, val_paths, train_labels, val_labels

    def train(self, image_paths: List[str], labels, 
              validation_split: float | None = None, progress_callback: Callable | None = None) -> Dict[str, Any]:
        """Train the model with optional progress callback (supports both single-label and multi-label)"""
        if len(image_paths) == 0:
            if progress_callback:
                progress_callback('error', 'No training data provided')
            print("No training data provided")
            return {}
        
        # Detect if we're in multi-label mode (labels are lists of floats)
        self.multi_label_mode = isinstance(labels[0], list)
        
        cfg = config_manager.hyperparameters
        if validation_split is None:
            validation_split = cfg.validation_split
        
        if progress_callback:
            progress_callback('status', 'Preparing training data...')
            progress_callback('detail', f'Total images: {len(image_paths)}')
            mode_str = 'Multi-label' if self.multi_label_mode else 'Single-label'
            progress_callback('detail', f'Training mode: {mode_str}')
        
        start_time = time.time()
        
        # Stratified split — balanced class representation in both sets
        if progress_callback:
            progress_callback('status', 'Splitting data (stratified)...')
        train_paths, val_paths, train_labels, val_labels = self._stratified_split(
            image_paths, labels, validation_split
        )
        
        if progress_callback:
            progress_callback('detail', f'Training set:   {len(train_paths)} images (stratified)')
            progress_callback('detail', f'Validation set: {len(val_paths)} images (stratified)')
        
        # Determine number of classes
        if self.multi_label_mode:
            num_classes = len(labels[0])  # Length of binary vector
            # Calculate tag frequency for balancing
            tag_counts = np.array(train_labels).sum(axis=0)
            total_samples = len(train_labels)
            
            if progress_callback:
                progress_callback('detail', f'Number of tags: {num_classes}')
                progress_callback('detail', '\n--- Tag Distribution (Top 10) ---')
                top_indices = np.argsort(tag_counts)[::-1][:10]
                for idx in top_indices:
                    tag_name = self.label_encoder.classes_[idx]
                    count = int(tag_counts[idx])
                    percentage = 100 * count / len(train_labels)
                    progress_callback('detail', f'  {tag_name}: {count} images ({percentage:.1f}%)')
                
                # Show imbalance ratio
                max_count = tag_counts.max()
                min_count = tag_counts[tag_counts > 0].min() if (tag_counts > 0).any() else 1
                imbalance_ratio = max_count / min_count if min_count > 0 else 1.0
                progress_callback('detail', f'Tag imbalance ratio: {imbalance_ratio:.2f}x')
                if imbalance_ratio > 3.0:
                    progress_callback('detail', '⚠ Significant tag imbalance detected - using pos_weight balancing')
        else:
            from collections import Counter
            class_counts = Counter(train_labels)
            num_classes = len(self.label_encoder.classes_)
            
            if progress_callback:
                progress_callback('detail', f'Number of classes: {num_classes}')
                progress_callback('detail', '\n--- Class Distribution ---')
                for class_idx, count in sorted(class_counts.items()):
                    class_name = self.label_encoder.classes_[class_idx]
                    percentage = 100 * count / len(train_labels)
                    progress_callback('detail', f'  {class_name}: {count} images ({percentage:.1f}%)')
        
        # Create datasets
        preprocessor = ImagePreprocessor()
        train_dataset = ImageDataset(train_paths, train_labels, preprocessor.transform, 
                                    multi_label=self.multi_label_mode)
        val_dataset = ImageDataset(val_paths, val_labels, preprocessor.transform,
                                  multi_label=self.multi_label_mode)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, 
                               shuffle=False, num_workers=0)
        
        # Initialize model
        if progress_callback:
            progress_callback('status', 'Initializing model...')
            progress_callback('detail', f'Model architecture: {config_manager.app_config.model_type}')
            progress_callback('detail', f'Output dimension: {num_classes}')
        
        self.model = ImageClassificationModel(num_classes=num_classes, pretrained=True)
        self.model.to(self.device)
        
        # Loss and optimizer (different for multi-label vs single-label)
        if self.multi_label_mode:
            # Binary Cross Entropy for multi-label with pos_weight for class balancing
            # Calculate positive weights: (# negative samples) / (# positive samples)
            tag_counts = np.array(train_labels).sum(axis=0)
            total_samples = len(train_labels)
            
            pos_weights = torch.zeros(num_classes)
            for tag_idx in range(num_classes):
                pos_count = tag_counts[tag_idx]
                neg_count = total_samples - pos_count
                if pos_count > 0:
                    # Weight = negative_count / positive_count
                    pos_weights[tag_idx] = neg_count / pos_count
                else:
                    pos_weights[tag_idx] = 1.0  # No positive examples, neutral weight
            
            pos_weights = pos_weights.to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            
            if progress_callback:
                progress_callback('detail', 'Loss function: BCEWithLogitsLoss with pos_weight balancing')
                max_weight = pos_weights.max().item()
                min_weight = pos_weights.min().item()
                progress_callback('detail', f'Pos weight range: {min_weight:.2f}x to {max_weight:.2f}x')
        else:
            # Calculate class weights for single-label
            class_counts = {}
            for label in train_labels:
                if isinstance(label, (list, np.ndarray)):
                    # Skip multi-label in single-label mode
                    continue
                class_counts[label] = class_counts.get(label, 0) + 1
            
            class_weights = torch.zeros(num_classes)
            total_samples = len(train_labels)
            for class_idx in range(num_classes):
                count = class_counts.get(class_idx, 1)
                class_weights[class_idx] = total_samples / (num_classes * count)
            class_weights = class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            if progress_callback:
                progress_callback('detail', 'Loss function: CrossEntropyLoss (single-label)')
        
        optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        
        if progress_callback:
            progress_callback('detail', f'Learning rate: {cfg.learning_rate}')
            progress_callback('detail', f'Batch size: {cfg.batch_size}')
            progress_callback('status', 'Starting training...')
        
        # Training loop
        best_accuracy = 0.0
        patience_counter = 0
        train_accuracies = []
        val_accuracies = []
        epoch_times: List[float] = []
        total_batches = len(train_loader)
        
        for epoch in range(cfg.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            epoch_start = time.time()
            
            for batch_idx, (images, batch_labels) in enumerate(train_loader):
                images = images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                
                if self.multi_label_mode:
                    # Multi-label accuracy: use sigmoid + threshold
                    predictions = torch.sigmoid(outputs) > 0.5
                    train_correct += (predictions == batch_labels.bool()).float().sum().item()
                    train_total += batch_labels.numel()  # Total number of labels
                else:
                    # Single-label accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
                
                # Per-batch progress update (every 5 batches or last batch)
                if progress_callback and total_batches > 0 and (batch_idx % 5 == 0 or batch_idx == total_batches - 1):
                    batch_pct = (batch_idx + 1) / total_batches
                    # Overall progress: 30% for registration, 70% for training
                    overall_pct = 30 + (epoch + batch_pct) / cfg.epochs * 70
                    progress_callback('progress', overall_pct)
                    progress_callback('status',
                        f'Epoch {epoch + 1}/{cfg.epochs} — batch {batch_idx + 1}/{total_batches}')
            
            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)
            train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
            
            # Validation
            val_accuracy = self.validate(val_loader)
            
            # Track both accuracies
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            self.training_metrics['losses'].append(avg_train_loss)
            self.training_metrics['train_accuracies'] = train_accuracies
            self.training_metrics['val_accuracies'] = val_accuracies
            self.training_metrics['accuracies'] = val_accuracies  # Keep for backward compatibility
            self.training_metrics['epochs_completed'] = epoch + 1
            
            # Calculate overfitting metric
            overfitting_gap = train_accuracy - val_accuracy
            
            # ETA estimation based on average epoch time so far
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            epochs_remaining = cfg.epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * epochs_remaining
            if eta_seconds >= 3600:
                eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
            elif eta_seconds >= 60:
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = f"{int(eta_seconds)}s"
            eta_label = f"ETA: {eta_str}" if epochs_remaining > 0 else "Done"
            
            epoch_msg = (f"Epoch {epoch+1}/{cfg.epochs} - "
                        f"Loss: {avg_train_loss:.4f}, "
                        f"Train Acc: {train_accuracy:.2f}%, "
                        f"Val Acc: {val_accuracy:.2f}%, "
                        f"{eta_label} (epoch took {epoch_elapsed:.1f}s)")
            print(epoch_msg)
            
            if progress_callback:
                progress_callback('status', f'Training: Epoch {epoch+1}/{cfg.epochs} — {eta_label}')
                progress_callback('detail', epoch_msg)
                progress_callback('metrics', {
                    'train_acc': train_accuracy,
                    'val_acc': val_accuracy,
                    'overfitting_gap': overfitting_gap
                })
                progress_callback('progress', 30 + ((epoch + 1) / cfg.epochs) * 70)
                
                # Overfitting/underfitting warnings
                if epoch >= 2:  # Check after a few epochs
                    if overfitting_gap > 15:
                        progress_callback('detail', f'⚠ Overfitting detected: train-val gap = {overfitting_gap:.1f}%')
                    elif train_accuracy < 50 and val_accuracy < 50:
                        progress_callback('detail', '⚠ Possible underfitting: both accuracies are low')
            
            # Early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        elapsed = time.time() - start_time
        self.training_metrics['total_time'] = elapsed
        
        # Save model
        if progress_callback:
            progress_callback('status', 'Saving model...')
        self.save_model('latest')
        
        if progress_callback:
            progress_callback('detail', f'Training completed in {elapsed:.2f} seconds')
            progress_callback('detail', f'Best validation accuracy: {best_accuracy:.2f}%')
            progress_callback('complete', True)
        
        return self.training_metrics
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model (handles both single-label and multi-label)"""
        if self.model is None:
            return 0.0
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                if self.multi_label_mode:
                    # Multi-label: sigmoid + threshold
                    predictions = torch.sigmoid(outputs) > 0.5
                    correct += (predictions == labels.bool()).float().sum().item()
                    total += labels.numel()
                else:
                    # Single-label: argmax
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy
    
    def save_model(self, model_name: str = 'latest'):
        """Save the trained model, training statistics and accuracy/loss graphs."""
        if self.model is None:
            print("No model to save")
            return

        model_type = config_manager.app_config.model_type

        model_state = {
            'model_state': self.model.state_dict(),
        }

        metadata = {
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'multi_label': self.multi_label_mode,
            'training_metrics': self.training_metrics,
            'hyperparameters': config_manager.hyperparameters.to_dict(),
            'model_type': model_type,
        }

        model_storage.save_model_weights(model_state, model_name, metadata)

        # --- Persist training statistics as JSON ---
        try:
            model_storage.save_training_stats(self.training_metrics, model_type)
        except Exception as exc:
            print(f"Warning: could not save training stats: {exc}")

        # --- Generate and save training graphs ---
        if _MATPLOTLIB_AVAILABLE:
            try:
                self._save_training_graphs(model_type)
            except Exception as exc:
                print(f"Warning: could not save training graphs: {exc}")

        # Log to database
        final_acc = self.training_metrics['accuracies'][-1] if self.training_metrics['accuracies'] else 0
        metadata_store.log_training_session(
            model_name=model_name,
            epochs=self.training_metrics['epochs_completed'],
            final_accuracy=final_acc,
            hyperparameters=config_manager.hyperparameters.to_dict(),
            notes=f"Training completed in {self.training_metrics['total_time']:.2f}s "
                  f"({'multi-label' if self.multi_label_mode else 'single-label'})"
        )

    def _save_training_graphs(self, model_type: str) -> None:
        """Generate accuracy and loss curves and persist them via model_storage."""
        epochs_done = self.training_metrics.get('epochs_completed', 0)
        if epochs_done == 0 or _plt is None:
            return

        x = list(range(1, epochs_done + 1))

        # ---- Accuracy graph ----
        train_accs = self.training_metrics.get('train_accuracies', [])
        val_accs   = self.training_metrics.get('val_accuracies',   [])
        if train_accs or val_accs:
            fig, ax = _plt.subplots(figsize=(8, 5))
            if train_accs:
                ax.plot(x[:len(train_accs)], train_accs,
                        label='Train Accuracy', marker='o', linewidth=2)
            if val_accs:
                ax.plot(x[:len(val_accs)], val_accs,
                        label='Validation Accuracy', marker='s', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Training Accuracy — {model_type}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            model_storage.save_training_graph(fig, 'accuracy', model_type)
            _plt.close(fig)

        # ---- Loss graph ----
        losses = self.training_metrics.get('losses', [])
        if losses:
            fig, ax = _plt.subplots(figsize=(8, 5))
            ax.plot(x[:len(losses)], losses,
                    label='Train Loss', color='red', marker='o', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss — {model_type}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            model_storage.save_training_graph(fig, 'loss', model_type)
            _plt.close(fig)

    # ------------------------------------------------------------------
    # Inbox fine-tuning
    # ------------------------------------------------------------------

    def fine_tune_from_annotations(
        self,
        annotations: Dict[str, List[str]],
        epochs: int = 5,
        lr: float = 0.0001,
        progress_callback: Callable | None = None,
    ) -> Dict[str, Any]:
        """Fine-tune the existing model on a small set of user-annotated inbox images.

        Loads the 'latest' model from disk if one is not already in memory.
        Only tags already known to the model are used so the architecture stays
        unchanged and catastrophic forgetting is minimised by the lower LR.

        Args:
            annotations: ``{absolute_image_path: [tag_name, ...]}``
            epochs:       Fine-tune epochs (default 5).
            lr:           Learning rate — kept small (default 1e-4) to avoid
                          overwriting full-training knowledge.
            progress_callback: Optional ``(event, value)`` callback.

        Returns:
            A dict of fine-tune metrics, or ``{}`` on failure.
        """
        # ---- Load model if not already in memory ---------------------------------
        if self.model is None:
            if progress_callback:
                progress_callback('status', 'Loading existing model weights…')
            try:
                model_state, metadata = model_storage.load_model_weights('latest')
                num_classes = metadata.get('num_classes', 10)
                self.multi_label_mode = metadata.get('multi_label', True)
                self.model = ImageClassificationModel(num_classes=num_classes)
                self.model.load_state_dict(model_state['model_state'])
                self.model.to(self.device)
                self.label_encoder.classes_ = np.array(metadata.get('classes', []))
            except Exception as exc:
                msg = f'Cannot load model for fine-tuning: {exc}'
                print(msg)
                if progress_callback:
                    progress_callback('error', msg)
                return {}

        known_classes = list(self.label_encoder.classes_)
        if not known_classes:
            msg = 'No known classes in model. Run a full training first.'
            print(msg)
            if progress_callback:
                progress_callback('error', msg)
            return {}

        # ---- Build training samples using the existing class vocabulary -----------
        self.multi_label_mode = True
        tag_to_idx = {tag: i for i, tag in enumerate(known_classes)}
        n_classes = len(known_classes)
        image_paths: List[str] = []
        multi_labels: List[List[float]] = []

        for img_path, tags in annotations.items():
            if not Path(img_path).exists():
                continue
            vec = [0.0] * n_classes
            matched = False
            for tag in tags:
                if tag in tag_to_idx:
                    vec[tag_to_idx[tag]] = 1.0
                    matched = True
            if matched:
                image_paths.append(img_path)
                multi_labels.append(vec)

        if not image_paths:
            msg = 'None of the annotated images have tags known to the model.'
            print(msg)
            if progress_callback:
                progress_callback('error', msg)
            return {}

        n = len(image_paths)
        if progress_callback:
            progress_callback('status',
                f'Fine-tuning on {n} inbox image(s) — {epochs} epoch(s)…')
            progress_callback('detail',
                f'Classes: {n_classes}  |  LR: {lr}  |  Epochs: {epochs}')

        # ---- Dataset / DataLoader ------------------------------------------------
        preprocessor = ImagePreprocessor()
        dataset = ImageDataset(image_paths, multi_labels,
                               preprocessor.transform, multi_label=True)
        loader = DataLoader(dataset,
                            batch_size=min(8, n),
                            shuffle=True, num_workers=0)

        # ---- Optimizer & loss (BCEWithLogits, no pos_weight for small sets) ------
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        start_time = time.time()
        ft_losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for images_batch, labels_batch in loader:
                images_batch = images_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            ft_losses.append(avg_loss)
            msg = f'Fine-tune epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}'
            print(msg)
            if progress_callback:
                progress_callback('status', msg)
                progress_callback('progress', 100 * (epoch + 1) / epochs)

        elapsed = time.time() - start_time
        self.model.eval()

        # ---- Persist updated weights ---------------------------------------------
        if progress_callback:
            progress_callback('status', 'Saving fine-tuned model…')
        self.save_model('latest')

        ft_metrics: Dict[str, Any] = {
            'losses': ft_losses,
            'accuracies': [],
            'epochs_completed': epochs,
            'total_time': elapsed,
            'n_images': n,
            'fine_tune': True,
        }
        if progress_callback:
            progress_callback('detail',
                f'Fine-tuning done in {elapsed:.1f}s  |  '
                f'Final loss: {ft_losses[-1]:.4f}')
            progress_callback('complete', True)

        return ft_metrics


class UnsupervisedClustering:
    """Unsupervised clustering for image sorting without labels"""
    
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.feature_extractor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from images using pretrained model"""
        # Use pretrained ResNet for feature extraction
        if self.feature_extractor is None:
            from torchvision.models import ResNet18_Weights
            self.feature_extractor = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.feature_extractor.fc = nn.Identity()  # type: ignore
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
        
        preprocessor = ImagePreprocessor()
        features_list = []
        
        for img_path in image_paths:
            tensor = preprocessor.preprocess_image(img_path)
            if tensor is not None:
                tensor = tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(tensor)
                features_list.append(features.cpu().numpy().flatten())
        
        return np.array(features_list)
    
    def cluster_images(self, image_paths: List[str]) -> Dict[int, List[str]]:
        """Cluster images and return groups"""
        print(f"Extracting features from {len(image_paths)} images...")
        features = self.extract_features(image_paths)
        
        print(f"Clustering into {self.n_clusters} groups...")
        cluster_labels = self.kmeans.fit_predict(features)
        
        # Group images by cluster
        clusters = {}
        for img_path, cluster_id in zip(image_paths, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(img_path)
        
        return clusters
