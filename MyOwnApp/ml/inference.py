"""InferenceEngine — loads a trained model and runs tag predictions."""

import time
import threading
from typing import Any, Dict, List

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import config_manager
from data_layer import model_storage
from .preprocessor import ImagePreprocessor
from .model import ImageClassificationModel


class TypedLabelEncoder(LabelEncoder):
    classes_: np.ndarray[Any, Any]

    def __init__(self) -> None:
        super().__init__()
        self.classes_ = np.array([], dtype=object)


class InferenceEngine:
    """Loads a saved model and predicts tags for single images or batches.

    Supports both single-label (softmax top-k) and multi-label (sigmoid
    threshold) inference modes, detected automatically from the saved metadata.
    """

    def __init__(self) -> None:
        self.model:            ImageClassificationModel | None = None
        self.label_encoder:    TypedLabelEncoder | None        = None
        self.multi_label_mode: bool                            = False
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config_manager.app_config.use_gpu
            else 'cpu'
        )
        self.preprocessor    = ImagePreprocessor()
        self.inference_times: List[float] = []
        # Guard model/encoder access across UI + background threads.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _infer_model_type_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> str | None:
        keys = list(state_dict.keys())
        if not keys:
            return None

        if any(key.startswith('backbone.conv1.') for key in keys):
            return 'resnet18'

        if any(key.startswith('backbone.features.') for key in keys):
            if any('.block.' in key for key in keys):
                return 'efficientnet_b0'
            if any('.conv.0.0.weight' in key for key in keys):
                return 'mobilenet_v2'

        if any(key.startswith('backbone.classifier.6.') for key in keys):
            return 'vgg16'

        return None

    def _candidate_model_types(self, metadata: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> List[str]:
        candidates: List[str] = []
        meta_type = metadata.get('model_type')
        cfg_type = config_manager.app_config.model_type
        inferred = self._infer_model_type_from_state_dict(state_dict)

        for model_type in (meta_type, cfg_type, inferred, 'efficientnet_b0', 'efficientnet_b1', 'mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'vgg16'):
            if isinstance(model_type, str) and model_type and model_type not in candidates:
                candidates.append(model_type)
        return candidates

    def load_model(self, model_name: str = 'latest') -> bool:
        with self._lock:
            try:
                model_state, metadata = model_storage.load_model_weights(model_name)
                num_classes            = metadata.get('num_classes', 10)
                self.multi_label_mode  = metadata.get('multi_label', False)
                state_dict             = model_state['model_state']

                last_error: Exception | None = None
                for model_type in self._candidate_model_types(metadata, state_dict):
                    try:
                        config_manager.app_config.model_type = model_type
                        self.model = ImageClassificationModel(num_classes=num_classes, pretrained=False)
                        self.model.load_state_dict(state_dict)
                        self.model.to(self.device)
                        self.model.eval()
                        self.label_encoder          = TypedLabelEncoder()
                        self.label_encoder.classes_ = np.array(metadata.get('classes', []))
                        metadata.setdefault('model_type', model_type)
                        print(f"Model loaded: {model_name} with {num_classes} classes ({'multi-label' if self.multi_label_mode else 'single-label'}) using {model_type}")
                        return True
                    except Exception as exc:
                        last_error = exc

                if last_error is not None:
                    raise last_error

            except Exception as exc:
                print(f"Error loading model: {exc}")
                return False

    # ------------------------------------------------------------------
    # Single-image prediction
    # ------------------------------------------------------------------

    def predict_single(self, image_path: str) -> List[Dict[str, Any]]:
        with self._lock:
            if self.model is None:
                return []
            t0     = time.time()
            tensor = self.preprocessor.preprocess_image(image_path)
            if tensor is None:
                return []
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs   = (torch.sigmoid(outputs) if self.multi_label_mode
                           else torch.softmax(outputs, dim=1))
            self.inference_times.append(time.time() - t0)
            return self._decode_probabilities(probs[0])

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, image_paths: List[str]) -> List[List[Dict[str, Any]]]:
        """Predict tags for every path; returns an empty list for failed images.

        Uses ``preprocess_batch``'s ``valid_indices`` to guarantee that
        results are always aligned 1-to-1 with *image_paths*, even when
        some images fail to load.
        """
        with self._lock:
            if self.model is None:
                return [[] for _ in image_paths]

            t0                    = time.time()
            batch_tensor, valid_i = self.preprocessor.preprocess_batch(image_paths)
            if batch_tensor is None:
                return [[] for _ in image_paths]

            batch_tensor = batch_tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs   = (torch.sigmoid(outputs) if self.multi_label_mode
                           else torch.softmax(outputs, dim=1))

            self.inference_times.append(time.time() - t0)

            results: List[List[Dict[str, Any]]] = [[] for _ in image_paths]
            for tensor_idx, orig_idx in enumerate(valid_i):
                results[orig_idx] = self._decode_probabilities(probs[tensor_idx])
            return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_probabilities(
        self, prob_dist: torch.Tensor
    ) -> List[Dict[str, Any]]:
        cfg         = config_manager.hyperparameters
        predictions: List[Dict[str, Any]] = []
        if self.label_encoder is None:
            return predictions

        if self.multi_label_mode:
            arr = prob_dist.cpu().numpy()
            for idx, prob in enumerate(arr):
                if prob >= cfg.confidence_threshold:
                    predictions.append({
                        'tag_name':   self.label_encoder.classes_[idx],
                        'confidence': float(prob),
                    })
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
        else:
            topk_p, topk_i = torch.topk(
                prob_dist, min(cfg.max_suggestions, len(prob_dist))
            )
            for prob, idx in zip(topk_p.cpu().numpy(), topk_i.cpu().numpy()):
                if prob >= cfg.confidence_threshold:
                    predictions.append({
                        'tag_name':   self.label_encoder.classes_[idx],
                        'confidence': float(prob),
                    })
        return predictions

    def get_avg_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
