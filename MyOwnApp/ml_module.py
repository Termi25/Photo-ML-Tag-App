"""
ml_module.py — backward-compatibility shim.

All ML components have been moved to the ``ml/`` package.
Any existing code that imports from this module continues to work unchanged.

    from ml_module import TrainingLoop, InferenceEngine   # still works
    from ml       import TrainingLoop, InferenceEngine    # new preferred style
"""

from ml import (  # noqa: F401
    ImagePreprocessor,
    ImageDataset,
    ImageClassificationModel,
    InferenceEngine,
    TrainingLoop,
    UnsupervisedClustering,
    _AMP_AVAILABLE,
    _SAFE_WORKERS,
    MATPLOTLIB_AVAILABLE as _MATPLOTLIB_AVAILABLE,
    _plt,
)

__all__ = [
    'ImagePreprocessor',
    'ImageDataset',
    'ImageClassificationModel',
    'InferenceEngine',
    'TrainingLoop',
    'UnsupervisedClustering',
    '_AMP_AVAILABLE',
    '_SAFE_WORKERS',
    '_MATPLOTLIB_AVAILABLE',
    '_plt',
]
