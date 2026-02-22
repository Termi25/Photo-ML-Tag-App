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
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image"""
        start_time = time.time()
        
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img)
            
            elapsed = time.time() - start_time
            self.preprocessing_times.append(elapsed)
            
            return tensor
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def preprocess_batch(self, image_paths: List[str]) -> torch.Tensor:
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
    """Custom dataset for image classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int] = None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image


class ImageClassificationModel(nn.Module):
    """Neural network model for image classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()
        cfg = config_manager.hyperparameters
        
        # Use ResNet18 as backbone
        if config_manager.app_config.model_type == 'resnet':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
        else:
            # Simple custom CNN
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
        in_features = num_features
        
        for hidden_size in cfg.hidden_layers:
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
    """Handles model inference for tag prediction"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                   config_manager.app_config.use_gpu else 'cpu')
        self.preprocessor = ImagePreprocessor()
        self.inference_times = []
    
    def load_model(self, model_name: str = 'latest'):
        """Load trained model"""
        try:
            model_state, metadata = model_storage.load_model_weights(model_name)
            
            num_classes = metadata.get('num_classes', 10)
            self.model = ImageClassificationModel(num_classes=num_classes)
            self.model.load_state_dict(model_state['model_state'])
            self.model.to(self.device)
            self.model.eval()
            
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(metadata.get('classes', []))
            
            print(f"Model loaded: {model_name} with {num_classes} classes")
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
            probabilities = torch.softmax(outputs, dim=1)
        
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        
        # Get top predictions
        cfg = config_manager.hyperparameters
        topk_prob, topk_indices = torch.topk(probabilities[0], 
                                             min(cfg.max_suggestions, len(probabilities[0])))
        
        predictions = []
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
            probabilities = torch.softmax(outputs, dim=1)
        
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        
        # Process predictions for each image
        batch_predictions = []
        cfg = config_manager.hyperparameters
        
        for prob_dist in probabilities:
            topk_prob, topk_indices = torch.topk(prob_dist, 
                                                 min(cfg.max_suggestions, len(prob_dist)))
            
            predictions = []
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
    """Handles model training with backpropagation"""
    
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
    
    def prepare_data_from_database(self) -> Tuple[List[str], List[int]]:
        """Prepare training data from database tags"""
        # Query all images with tags
        cursor = metadata_store.connection.cursor()
        cursor.execute('''
            SELECT i.file_path, GROUP_CONCAT(t.tag_name) as tags
            FROM images i
            JOIN image_tags it ON i.id = it.image_id
            JOIN tags t ON it.tag_id = t.id
            GROUP BY i.id
            HAVING tags IS NOT NULL
        ''')
        
        image_paths = []
        tag_lists = []
        
        for row in cursor.fetchall():
            image_paths.append(row[0])
            # Use first tag as primary label (for simplicity)
            tags = row[1].split(',')
            tag_lists.append(tags[0] if tags else 'unknown')
        
        if len(image_paths) == 0:
            print("No training data found in database")
            return [], []
        
        # Encode labels
        labels = self.label_encoder.fit_transform(tag_lists)
        
        print(f"Prepared {len(image_paths)} images with {len(self.label_encoder.classes_)} classes")
        return image_paths, labels.tolist()
    
    def prepare_data_from_folders(self, root_folder: str) -> Tuple[List[str], List[int]]:
        """Bootstrap training data from folder structure"""
        root_path = Path(root_folder)
        image_paths = []
        labels = []
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif'}
        
        # Traverse folders
        for folder in root_path.rglob('*'):
            if folder.is_dir():
                folder_name = folder.name
                
                for img_file in folder.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        image_paths.append(str(img_file))
                        labels.append(folder_name)
        
        if len(image_paths) == 0:
            print("No images found in folder structure")
            return [], []
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"Bootstrapped {len(image_paths)} images from {len(self.label_encoder.classes_)} folders")
        return image_paths, encoded_labels.tolist()
    
    def train(self, image_paths: List[str], labels: List[int], 
              validation_split: float = None, progress_callback: Callable = None) -> Dict[str, Any]:
        """Train the model with optional progress callback"""
        if len(image_paths) == 0:
            if progress_callback:
                progress_callback('error', 'No training data provided')
            print("No training data provided")
            return {}
        
        cfg = config_manager.hyperparameters
        if validation_split is None:
            validation_split = cfg.validation_split
        
        if progress_callback:
            progress_callback('status', 'Preparing training data...')
            progress_callback('detail', f'Total images: {len(image_paths)}')
        
        start_time = time.time()
        
        # Split data
        split_idx = int(len(image_paths) * (1 - validation_split))
        train_paths, val_paths = image_paths[:split_idx], image_paths[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        if progress_callback:
            progress_callback('detail', f'Training set: {len(train_paths)} images')
            progress_callback('detail', f'Validation set: {len(val_paths)} images')
            progress_callback('detail', f'Number of classes: {len(set(labels))}')
        
        # Calculate class distribution and weights for balancing
        from collections import Counter
        class_counts = Counter(train_labels)
        num_classes = len(self.label_encoder.classes_)
        
        if progress_callback:
            progress_callback('detail', '\n--- Class Distribution ---')
            for class_idx, count in sorted(class_counts.items()):
                class_name = self.label_encoder.classes_[class_idx]
                percentage = 100 * count / len(train_labels)
                progress_callback('detail', f'  {class_name}: {count} images ({percentage:.1f}%)')
        
        # Calculate class weights (inverse frequency)
        class_weights = torch.zeros(num_classes)
        total_samples = len(train_labels)
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            class_weights[class_idx] = total_samples / (num_classes * count)
        class_weights = class_weights.to(self.device)
        
        if progress_callback:
            max_weight = class_weights.max().item()
            min_weight = class_weights.min().item()
            imbalance_ratio = max_weight / min_weight if min_weight > 0 else 1.0
            progress_callback('detail', f'Class imbalance ratio: {imbalance_ratio:.2f}x')
            if imbalance_ratio > 3.0:
                progress_callback('detail', '⚠ Significant class imbalance detected - using weighted loss')
        
        # Create datasets
        preprocessor = ImagePreprocessor()
        train_dataset = ImageDataset(train_paths, train_labels, preprocessor.transform)
        val_dataset = ImageDataset(val_paths, val_labels, preprocessor.transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, 
                               shuffle=False, num_workers=0)
        
        # Initialize model
        if progress_callback:
            progress_callback('status', 'Initializing model...')
            progress_callback('detail', f'Model architecture: {config_manager.app_config.model_type}')
            progress_callback('detail', f'Number of output classes: {num_classes}')
        
        self.model = ImageClassificationModel(num_classes=num_classes, pretrained=True)
        self.model.to(self.device)
        
        # Loss and optimizer with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        
        for epoch in range(cfg.epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, batch_labels in train_loader:
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
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
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
            self.training_metrics['epochs_completed'] = epoch + 1
            
            # Calculate overfitting metric
            overfitting_gap = train_accuracy - val_accuracy
            
            epoch_msg = (f"Epoch {epoch+1}/{cfg.epochs} - "
                        f"Loss: {avg_train_loss:.4f}, "
                        f"Train Acc: {train_accuracy:.2f}%, "
                        f"Val Acc: {val_accuracy:.2f}%")
            print(epoch_msg)
            
            if progress_callback:
                progress_callback('status', f'Training: Epoch {epoch+1}/{cfg.epochs}')
                progress_callback('detail', epoch_msg)
                progress_callback('metrics', {
                    'train_acc': train_accuracy,
                    'val_acc': val_accuracy,
                    'overfitting_gap': overfitting_gap
                })
                progress_callback('progress', ((epoch + 1) / cfg.epochs) * 100)
                
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
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        return accuracy
    
    def save_model(self, model_name: str = 'latest'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return
        
        model_state = {
            'model_state': self.model.state_dict(),
        }
        
        metadata = {
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'training_metrics': self.training_metrics,
            'hyperparameters': config_manager.hyperparameters.to_dict()
        }
        
        model_storage.save_model_weights(model_state, model_name, metadata)
        
        # Log to database
        metadata_store.log_training_session(
            model_name=model_name,
            epochs=self.training_metrics['epochs_completed'],
            final_accuracy=self.training_metrics['accuracies'][-1] if self.training_metrics['accuracies'] else 0,
            hyperparameters=config_manager.hyperparameters.to_dict(),
            notes=f"Training completed in {self.training_metrics['total_time']:.2f}s"
        )


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
            self.feature_extractor = models.resnet18(pretrained=True)
            self.feature_extractor.fc = nn.Identity()
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
