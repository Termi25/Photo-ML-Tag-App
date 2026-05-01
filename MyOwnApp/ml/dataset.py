"""PyTorch Dataset wrapper for image classification (single-label and multi-label)."""

from typing import List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

from config import config_manager


class ImageDataset(Dataset):
    """Custom dataset supporting both single-label and multi-label classification.

    Parameters
    ----------
    image_paths : absolute paths to image files.
    labels      : ``List[int]`` for single-label, ``List[List[float]]`` for multi-label,
                  or ``None`` when no labels are available (inference-only use).
    transform   : torchvision transform pipeline to apply.
    multi_label : when ``True`` labels are treated as binary vectors.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[Union[List[int], List[List[float]]]] = None,
        transform=None,
        multi_label: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform
        self.multi_label = multi_label

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a black placeholder so DataLoader never hangs on corrupt files.
            cfg   = config_manager.hyperparameters
            image = Image.new('RGB', cfg.image_size, (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            if self.multi_label:
                return image, torch.FloatTensor(self.labels[idx])  # type: ignore[index]
            else:
                return image, self.labels[idx]  # type: ignore[index]
        return image
