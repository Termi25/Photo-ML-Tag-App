"""Image preprocessing — inference and training transforms."""

import time
from typing import List, Optional, Tuple

import torch
from torchvision import transforms
from PIL import Image

from config import config_manager


class ImagePreprocessor:
    """Handles image preprocessing for both training and inference.

    Two transform pipelines:
    - ``transform``       — deterministic (Resize → CenterCrop): used for inference and validation.
    - ``train_transform`` — stochastic (RandomResizedCrop + colour jitter): used for training.
    """

    def __init__(self) -> None:
        cfg  = config_manager.hyperparameters
        self.image_size = cfg.image_size
        h, w = self.image_size
        resize_to = int(max(h, w) * 1.143)   # e.g. 256 for a 224-target

        self.transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.normalization_mean, std=cfg.normalization_std),
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.normalization_mean, std=cfg.normalization_std),
        ])

        self.preprocessing_times: List[float] = []

    # ------------------------------------------------------------------

    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess a single image for inference. Returns ``None`` on failure."""
        t0 = time.time()
        try:
            img    = Image.open(image_path).convert('RGB')
            tensor = self.transform(img)
            self.preprocessing_times.append(time.time() - t0)
            return tensor  # type: ignore[return-value]
        except Exception as exc:
            print(f"Error preprocessing {image_path}: {exc}")
            return None

    def preprocess_batch(
        self, image_paths: List[str]
    ) -> Tuple[Optional[torch.Tensor], List[int]]:
        """Return ``(stacked_tensor, valid_indices)``.

        ``valid_indices[i]`` is the original position in *image_paths* of the
        i-th row in the returned tensor.  Images that fail preprocessing are
        omitted from the tensor so callers can emit empty predictions for them
        without producing off-by-N misalignments.
        """
        tensors: List[torch.Tensor] = []
        valid:   List[int]          = []
        for idx, path in enumerate(image_paths):
            t = self.preprocess_image(path)
            if t is not None:
                tensors.append(t)
                valid.append(idx)
        if not tensors:
            return None, []
        return torch.stack(tensors), valid

    def get_avg_preprocessing_time(self) -> float:
        if not self.preprocessing_times:
            return 0.0
        return sum(self.preprocessing_times) / len(self.preprocessing_times)
