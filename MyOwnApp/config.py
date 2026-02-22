"""
Configuration Module
Defines hyperparameters and configuration settings for the ML-based image sorting application.
These parameters can be modified through the UI without changing code.
"""

import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class HyperParameters:
    """Hyperparameters for ML model training and inference"""
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    dropout_rate: float = 0.3
    
    # Model architecture
    embedding_dim: int = 128
    hidden_layers: list = None
    activation: str = 'relu'
    
    # Training behavior
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    
    # Preprocessing
    image_size: tuple = (224, 224)
    normalization_mean: tuple = (0.485, 0.456, 0.406)
    normalization_std: tuple = (0.229, 0.224, 0.225)
    
    # Inference
    confidence_threshold: float = 0.5
    max_suggestions: int = 5
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert tuples to lists for JSON serialization
        data['image_size'] = list(data['image_size'])
        data['normalization_mean'] = list(data['normalization_mean'])
        data['normalization_std'] = list(data['normalization_std'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperParameters':
        """Create from dictionary"""
        # Convert lists back to tuples where needed
        if 'image_size' in data:
            data['image_size'] = tuple(data['image_size'])
        if 'normalization_mean' in data:
            data['normalization_mean'] = tuple(data['normalization_mean'])
        if 'normalization_std' in data:
            data['normalization_std'] = tuple(data['normalization_std'])
        return cls(**data)


@dataclass
class ApplicationConfig:
    """General application configuration"""
    # Paths
    default_image_folder: str = "../ExampleImageFolder"
    database_path: str = "./data/image_metadata.db"
    model_weights_dir: str = "./data/models"
    cache_dir: str = "./data/cache"
    
    # UI settings
    thumbnail_size: tuple = (120, 120)
    gallery_columns: int = 5
    theme: str = "default"
    
    # ML settings
    use_gpu: bool = True
    num_workers: int = 4
    model_type: str = "resnet"  # Options: resnet, efficientnet, custom
    
    # Training mode
    supervised_mode: bool = True  # If False, uses unsupervised clustering
    bootstrap_from_folders: bool = True  # Use folder structure as initial labels
    
    # Performance
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['thumbnail_size'] = list(data['thumbnail_size'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationConfig':
        """Create from dictionary"""
        if 'thumbnail_size' in data:
            data['thumbnail_size'] = tuple(data['thumbnail_size'])
        return cls(**data)


class ConfigManager:
    """Manages loading and saving of configuration"""
    
    def __init__(self, config_path: str = "./config.json"):
        self.config_path = Path(config_path)
        self.hyperparameters = HyperParameters()
        self.app_config = ApplicationConfig()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                if 'hyperparameters' in data:
                    self.hyperparameters = HyperParameters.from_dict(data['hyperparameters'])
                
                if 'app_config' in data:
                    self.app_config = ApplicationConfig.from_dict(data['app_config'])
                
                print(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        else:
            print("No config file found. Using default configuration.")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'hyperparameters': self.hyperparameters.to_dict(),
                'app_config': self.app_config.to_dict()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def update_hyperparameters(self, **kwargs):
        """Update hyperparameters"""
        for key, value in kwargs.items():
            if hasattr(self.hyperparameters, key):
                setattr(self.hyperparameters, key, value)
        self.save_config()
    
    def update_app_config(self, **kwargs):
        """Update application configuration"""
        for key, value in kwargs.items():
            if hasattr(self.app_config, key):
                setattr(self.app_config, key, value)
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset all configurations to defaults"""
        self.hyperparameters = HyperParameters()
        self.app_config = ApplicationConfig()
        self.save_config()


# Global configuration instance
config_manager = ConfigManager()
