"""UnsupervisedClustering — KMeans clustering over pretrained ResNet18 features."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from .preprocessor import ImagePreprocessor


class UnsupervisedClustering:
    """Groups images into *n_clusters* clusters using ResNet18 feature vectors.

    Useful when no labels exist: cluster IDs are written back as tags so the
    user can inspect groups and assign meaningful names later.
    """

    def __init__(self, n_clusters: int = 10) -> None:
        self.n_clusters       = n_clusters
        self.kmeans           = KMeans(n_clusters=n_clusters, random_state=42)
        self.feature_extractor = None
        self.device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Return an (N, D) array of backbone features for each image."""
        if self.feature_extractor is None:
            from torchvision import models
            from torchvision.models import ResNet18_Weights
            fe        = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            fe.fc     = nn.Identity()  # type: ignore[assignment]
            fe.to(self.device)
            fe.eval()
            self.feature_extractor = fe

        preprocessor  = ImagePreprocessor()
        features_list: List[np.ndarray] = []

        for img_path in image_paths:
            tensor = preprocessor.preprocess_image(img_path)
            if tensor is not None:
                tensor = tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.feature_extractor(tensor)
                features_list.append(feat.cpu().numpy().flatten())

        return np.array(features_list)

    def cluster_images(self, image_paths: List[str]) -> Dict[int, List[str]]:
        """Cluster *image_paths* and return ``{cluster_id: [path, ...]}``."""
        print(f"Extracting features from {len(image_paths)} images...")
        features = self.extract_features(image_paths)

        if len(features) == 0:
            return {}

        # KMeans requires n_clusters ≤ n_samples
        actual = min(self.n_clusters, len(features))
        if actual != self.n_clusters:
            print(f"Reduced clusters from {self.n_clusters} to {actual} (not enough samples)")
            self.kmeans = KMeans(n_clusters=actual, random_state=42)

        print(f"Clustering into {actual} groups...")
        labels = self.kmeans.fit_predict(features)

        clusters: Dict[int, List[str]] = {}
        for path, cid in zip(image_paths, labels):
            clusters.setdefault(int(cid), []).append(path)
        return clusters
