"""
ml/ — Machine learning package.

Package layout
--------------
_constants.py     — module-level flags (AMP, workers, matplotlib) — no ml imports
preprocessor.py   — ImagePreprocessor (inference + training transforms)
dataset.py        — ImageDataset (single-label & multi-label)
model.py          — ImageClassificationModel (ResNet / EfficientNet / VGG / MobileNet)
inference.py      — InferenceEngine
training.py       — TrainingLoop (supervised training + fine-tuning)
clustering.py     — UnsupervisedClustering (KMeans)

Import either style:
    from ml import TrainingLoop           # clean package import
    from ml_module import TrainingLoop    # backward-compat shim
"""

# ── Constants (imported first so sub-modules can safely import from _constants) ─
from ._constants import (
    AMP_AVAILABLE, _AMP_AVAILABLE,
    SAFE_WORKERS,  _SAFE_WORKERS,
    MATPLOTLIB_AVAILABLE, _MATPLOTLIB_AVAILABLE,
    _plt,
)

# ── Public components ────────────────────────────────────────────────────────
from .preprocessor import ImagePreprocessor
from .dataset      import ImageDataset
from .model        import ImageClassificationModel
from .inference    import InferenceEngine
from .training     import TrainingLoop
from .clustering   import UnsupervisedClustering

__all__ = [
    # classes
    'ImagePreprocessor',
    'ImageDataset',
    'ImageClassificationModel',
    'InferenceEngine',
    'TrainingLoop',
    'UnsupervisedClustering',
    # constants (both aliases)
    'AMP_AVAILABLE',  '_AMP_AVAILABLE',
    'SAFE_WORKERS',   '_SAFE_WORKERS',
    'MATPLOTLIB_AVAILABLE', '_MATPLOTLIB_AVAILABLE',
]
