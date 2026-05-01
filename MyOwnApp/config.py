"""
Configuration Module
Defines hyperparameters and configuration settings for the ML-based image sorting application.
These parameters can be modified through the UI without changing code.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    hidden_layers: list | None = None
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
    model_type: str = "resnet18"  # Options: resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b1, mobilenet_v2, vgg16
    auto_load_model: bool = False  # If True, attempt to auto-load latest model at startup
    
    # Training mode
    supervised_mode: bool = True  # If False, uses unsupervised clustering
    bootstrap_from_folders: bool = True  # Use folder structure as initial labels
    
    # Tag settings
    tag_root_folder: str = ""   # Root folder for folder-based tag extraction (empty = uses default_image_folder)
    auto_folder_tags: bool = False  # Automatically add folder-path tags when registering images
    
    # Database settings
    db_root_folder: str = ""    # Root folder for the single shared database (empty = uses app ./data/ dir)
    
    # Benchmarking — COCO dataset
    coco_data_dir: str = ""    # local cache for annotations + images (empty = ./data/coco)
    coco_max_images: int = 500  # how many val2017 images to use per benchmark run

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


@dataclass
class TrainingProfile:
    """Named snapshot of all training-relevant hyperparameters + architecture."""
    name: str
    model_type: str              = 'resnet18'
    learning_rate: float         = 0.0001
    batch_size: int              = 32
    epochs: int                  = 20
    dropout_rate: float          = 0.3
    early_stopping_patience: int = 7
    validation_split: float      = 0.15
    confidence_threshold: float  = 0.35
    max_suggestions: int         = 5
    image_size: int              = 224
    builtin: bool                = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingProfile':
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


_BUILTIN_PROFILES: List[TrainingProfile] = [
    TrainingProfile(
        name='Fast', model_type='mobilenet_v2',
        learning_rate=0.001, batch_size=32, epochs=5,
        dropout_rate=0.3, early_stopping_patience=2,
        validation_split=0.2, confidence_threshold=0.4,
        builtin=True,
    ),
    TrainingProfile(
        name='Balanced', model_type='efficientnet_b0',
        learning_rate=0.0001, batch_size=16, epochs=20,
        dropout_rate=0.3, early_stopping_patience=7,
        validation_split=0.15, confidence_threshold=0.35,
        builtin=True,
    ),
    TrainingProfile(
        name='High Quality', model_type='resnet50',
        learning_rate=0.0001, batch_size=16, epochs=30,
        dropout_rate=0.25, early_stopping_patience=10,
        validation_split=0.15, confidence_threshold=0.35,
        builtin=True,
    ),
    TrainingProfile(
        name='Paper — ResNet-18', model_type='resnet18',
        learning_rate=0.0001, batch_size=16, epochs=25,
        dropout_rate=0.35, early_stopping_patience=8,
        validation_split=0.15, confidence_threshold=0.35,
        builtin=True,
    ),
    TrainingProfile(
        name='Paper — EfficientNet-B0', model_type='efficientnet_b0',
        learning_rate=0.0001, batch_size=16, epochs=25,
        dropout_rate=0.35, early_stopping_patience=8,
        validation_split=0.15, confidence_threshold=0.35,
        builtin=True,
    ),
]


class ProfileManager:
    """Manages named training profiles: built-ins (read-only) + user-created."""

    def __init__(self, profiles_path: str = './profiles.json') -> None:
        self._path = Path(profiles_path)
        # Ordered: built-ins first, then user profiles
        self._profiles: Dict[str, TrainingProfile] = {
            p.name: p for p in _BUILTIN_PROFILES
        }
        self._load_user()

    def _load_user(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, 'r') as f:
                data = json.load(f)
            for entry in data.get('profiles', []):
                try:
                    p = TrainingProfile.from_dict(entry)
                    p.builtin = False
                    self._profiles[p.name] = p
                except Exception:
                    pass
        except Exception as e:
            print(f'Could not load profiles: {e}')

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            user = [p.to_dict() for p in self._profiles.values() if not p.builtin]
            with open(self._path, 'w') as f:
                json.dump({'profiles': user}, f, indent=2)
        except Exception as e:
            print(f'Could not save profiles: {e}')

    def list_names(self) -> List[str]:
        return list(self._profiles.keys())

    def get(self, name: str) -> Optional[TrainingProfile]:
        return self._profiles.get(name)

    def save(self, profile: TrainingProfile) -> None:
        profile.builtin = False
        self._profiles[profile.name] = profile
        self._persist()

    def delete(self, name: str) -> bool:
        p = self._profiles.get(name)
        if p is None or p.builtin:
            return False
        del self._profiles[name]
        self._persist()
        return True

    def is_builtin(self, name: str) -> bool:
        p = self._profiles.get(name)
        return p is not None and p.builtin


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

# Global profile manager instance
profile_manager = ProfileManager()
