"""TrainingLoop — full supervised training with AMP, stratified splits, and fine-tuning."""

import gc
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from config import config_manager
from data_layer import metadata_store, model_storage
from ._constants import (
    AMP_AVAILABLE as _AMP_AVAILABLE,
    SAFE_WORKERS  as _SAFE_WORKERS,
    MATPLOTLIB_AVAILABLE as _MATPLOTLIB_AVAILABLE,
    _plt,
)
from .preprocessor import ImagePreprocessor
from .dataset      import ImageDataset
from .model        import ImageClassificationModel


class TypedLabelEncoder(LabelEncoder):
    classes_: np.ndarray[Any, Any]

    def __init__(self) -> None:
        super().__init__()
        # Ensure the attribute always exists (COCO benchmark doesn't call .fit()).
        self.classes_ = np.array([], dtype=object)


class TrainingLoop:
    """Full supervised training pipeline.

    Supports both **single-label** (CrossEntropyLoss) and **multi-label**
    (BCEWithLogitsLoss + pos_weight balancing) classification.
    Uses AMP on CUDA for ~2× throughput and iterative stratified splits for
    balanced train/val sets even with highly imbalanced tag distributions.
    """

    def __init__(self) -> None:
        self.model:            Optional[ImageClassificationModel] = None
        self.label_encoder     = TypedLabelEncoder()
        self.device            = torch.device(
            'cuda' if torch.cuda.is_available() and config_manager.app_config.use_gpu
            else 'cpu'
        )
        self.training_metrics: Dict[str, Any] = {
            'losses': [], 'accuracies': [], 'epochs_completed': 0, 'total_time': 0.0,
            'val_macro_f1': [], 'val_micro_f1': [],
            'train_accuracies': [], 'val_accuracies': [],
        }
        self.multi_label_mode = False
        self.data_source: str = ''
        # Held-out test split (populated when test_split > 0 in train())
        self.test_paths:  List[str] = []
        self.test_labels: list      = []
        # Last validation predictions — used for confusion matrix generation.
        self._last_val_preds:  Optional[np.ndarray] = None
        self._last_val_labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data_from_database(
        self, progress_callback: Optional[Callable] = None
    ) -> Tuple[List[str], List[List[float]]]:
        """Load multi-label training data from all database tags."""
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

        if progress_callback:
            progress_callback('status', f'Verifying {len(rows)} image paths...')

        image_paths:        List[str]        = []
        all_tags_per_image: List[List[str]]  = []
        skipped = 0
        for row in rows:
            if not Path(row[0]).exists():
                skipped += 1
                continue
            tags = [t.strip() for t in row[1].split('|||') if t.strip()]
            if tags:
                image_paths.append(row[0])
                all_tags_per_image.append(tags)

        if skipped and progress_callback:
            progress_callback('detail', f'Skipped {skipped} files no longer on disk')
        if not image_paths:
            print("No training data found in database")
            return [], []

        all_unique_tags: set = set()
        for tags in all_tags_per_image:
            all_unique_tags.update(tags)
        sorted_tags = sorted(all_unique_tags)
        self.label_encoder.classes_ = np.array(sorted_tags)
        tag_to_idx = {t: i for i, t in enumerate(sorted_tags)}

        multi_labels: List[List[float]] = []
        for tags in all_tags_per_image:
            vec = [0.0] * len(sorted_tags)
            for t in tags:
                vec[tag_to_idx[t]] = 1.0
            multi_labels.append(vec)

        print(f"Prepared {len(image_paths)} images with {len(sorted_tags)} unique tags (multi-label)")
        if progress_callback:
            progress_callback('detail', f'Images: {len(image_paths)}, Tags: {len(sorted_tags)}')
            progress_callback('progress', 5)
        return image_paths, multi_labels

    def prepare_data_from_folders(
        self, root_folder: str, progress_callback: Optional[Callable] = None
    ) -> Tuple[List[str], List[List[float]]]:
        """Bootstrap training data from folder structure using hierarchical multi-label tags."""
        root_path          = Path(root_folder)
        image_paths:        List[str]       = []
        all_tags_per_image: List[List[str]] = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif'}

        print("Scanning folders and extracting hierarchical tags...")
        if progress_callback:
            progress_callback('status', 'Scanning folder structure...')

        for img_file in root_path.rglob('*'):
            if not (img_file.is_file() and img_file.suffix.lower() in image_extensions):
                continue
            if img_file.name.startswith('.'):
                continue
            relative_path  = img_file.relative_to(root_path)
            parent_folders = list(relative_path.parent.parts)
            if '_INBOX' in parent_folders:
                continue
            if any(p.lower() == 'thumbs' for p in parent_folders):
                continue
            if parent_folders and parent_folders[0] != '.':
                image_paths.append(str(img_file))
                all_tags_per_image.append(parent_folders)

        if not image_paths:
            print("No images found in folder structure")
            return [], []

        all_unique_tags: set = set()
        for tags in all_tags_per_image:
            all_unique_tags.update(tags)
        sorted_tags = sorted(all_unique_tags)
        self.label_encoder.classes_ = np.array(sorted_tags)
        tag_to_idx = {t: i for i, t in enumerate(sorted_tags)}

        print(f"Found {len(sorted_tags)} unique tags: "
              f"{', '.join(sorted_tags[:10])}{'...' if len(sorted_tags) > 10 else ''}")

        multi_labels: List[List[float]] = []
        for tags in all_tags_per_image:
            vec = [0.0] * len(sorted_tags)
            for t in tags:
                vec[tag_to_idx[t]] = 1.0
            multi_labels.append(vec)

        # Register images + tags in DB (single bulk transaction)
        from data_layer import metadata_store as _ms
        from logic_controller import TaggingEngine
        tagging_engine = TaggingEngine()
        total = len(image_paths)
        if progress_callback:
            progress_callback('status', f'Registering {total} images in database...')

        try:
            _ms.connection.execute("BEGIN")
            for reg_idx, (img_path, tags) in enumerate(zip(image_paths, all_tags_per_image)):
                p        = Path(img_path)
                image_id = _ms.add_image(
                    file_path=img_path, file_name=p.name,
                    file_size=p.stat().st_size if p.exists() else 0, commit=False,
                )
                if image_id is not None:
                    for tag in tags:
                        tag_id = _ms.get_or_create_tag(tag, category='folder_supervised', commit=False)
                        _ms.add_image_tag(
                            image_id=image_id, tag_id=tag_id,
                            confidence=1.0, source='folder_supervised', commit=False,
                        )
                try:
                    tagging_engine.write_tags_to_file(img_path, tags, silent=True)
                except Exception:
                    pass

                if progress_callback and (reg_idx % 10 == 0 or reg_idx == total - 1):
                    pct = int(100 * (reg_idx + 1) / total)
                    progress_callback('status', f'Registering images: {reg_idx + 1}/{total} ({pct}%)')
                    progress_callback('progress', pct * 0.3)
            _ms.connection.commit()
        except Exception:
            _ms.connection.rollback()
            raise

        for i in range(min(3, len(image_paths))):
            print(f"  {Path(image_paths[i]).name} -> [{', '.join(all_tags_per_image[i])}]")
        print(f"\nBootstrapped {len(image_paths)} images with {len(sorted_tags)} unique tags (multi-label)")
        return image_paths, multi_labels

    # ------------------------------------------------------------------
    # Stratified split
    # ------------------------------------------------------------------

    def _stratified_split(
        self, image_paths: List[str], labels: list, validation_split: float
    ) -> Tuple[List[str], List[str], list, list]:
        n = len(image_paths)
        if self.multi_label_mode:
            label_matrix = np.array(labels, dtype=np.float32)
            n_tags       = label_matrix.shape[1]
            tag_counts   = label_matrix.sum(axis=0)
            desired_val  = tag_counts * validation_split
            rng          = np.random.default_rng(42)
            indices      = rng.permutation(n).tolist()
            val_size     = max(1, int(round(n * validation_split)))
            val_tag_counts = np.zeros(n_tags, dtype=np.float32)
            train_idx: List[int] = []
            val_idx:   List[int] = []
            for idx in indices:
                active = np.where(label_matrix[idx] == 1)[0]
                if len(active) == 0:
                    (val_idx if len(val_idx) < val_size else train_idx).append(idx)
                    continue
                deficit = desired_val[active] - val_tag_counts[active]
                if len(val_idx) < val_size and deficit.max() > 0:
                    val_idx.append(idx)
                    val_tag_counts[active] += 1
                else:
                    train_idx.append(idx)
            return (
                [image_paths[i] for i in train_idx], [image_paths[i] for i in val_idx],
                [labels[i]      for i in train_idx], [labels[i]      for i in val_idx],
            )
        else:
            from sklearn.model_selection import train_test_split
            try:
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    image_paths, labels, test_size=validation_split,
                    stratify=labels, random_state=42,
                )
                return train_paths, val_paths, train_labels, val_labels
            except ValueError:
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    image_paths, labels, test_size=validation_split, random_state=42,
                )
                return train_paths, val_paths, train_labels, val_labels

    def _three_way_split(
        self,
        image_paths: List[str],
        labels: list,
        val_split: float,
        test_split: float,
    ) -> Tuple[List[str], List[str], List[str], list, list, list]:
        """Stratified 70/15/15-style split into train, val, and held-out test sets.

        Returns (train_paths, val_paths, test_paths,
                 train_labels, val_labels, test_labels).
        """
        # First carve out the test set from the full dataset
        train_val_paths, test_paths, train_val_labels, test_labels = self._stratified_split(
            image_paths, labels, test_split
        )
        # Then split the remaining data into train and val
        # val fraction relative to the train+val pool
        adjusted_val = val_split / (1.0 - test_split) if (1.0 - test_split) > 0 else val_split
        adjusted_val = min(adjusted_val, 0.5)
        train_paths, val_paths, train_labels, val_labels = self._stratified_split(
            train_val_paths, train_val_labels, adjusted_val
        )
        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        image_paths: List[str],
        labels,
        validation_split: Optional[float] = None,
        test_split: float = 0.0,
        progress_callback: Optional[Callable] = None,
        cancel_event=None,
    ) -> Dict[str, Any]:
        if not image_paths:
            if progress_callback:
                progress_callback('error', 'No training data provided')
            return {}

        self.multi_label_mode = isinstance(labels[0], list)
        cfg = config_manager.hyperparameters
        if validation_split is None:
            validation_split = cfg.validation_split

        # COCO benchmark feeds pre-vectorized multi-label targets; in that path
        # LabelEncoder is never fit, but we still need stable class names.
        if self.multi_label_mode:
            try:
                num_classes_hint = len(labels[0])
            except Exception:
                num_classes_hint = 0

            if (not hasattr(self.label_encoder, 'classes_')
                    or len(getattr(self.label_encoder, 'classes_', [])) != num_classes_hint):
                # Try to recover class names from DB tags (COCO loader seeds category='coco').
                try:
                    cursor = metadata_store.connection.cursor()
                    cursor.execute(
                        "SELECT tag_name FROM tags WHERE category = ? ORDER BY tag_name",
                        ('coco',)
                    )
                    coco_names = [r[0] for r in cursor.fetchall()]
                except Exception:
                    coco_names = []

                if coco_names and len(coco_names) == num_classes_hint:
                    self.label_encoder.classes_ = np.array(coco_names, dtype=object)
                else:
                    self.label_encoder.classes_ = np.array(
                        [f'class_{i}' for i in range(num_classes_hint)],
                        dtype=object,
                    )

        if progress_callback:
            progress_callback('status', 'Preparing training data...')
            progress_callback('detail', f'Total images: {len(image_paths)}')
            progress_callback('detail', f'Training mode: {"Multi-label" if self.multi_label_mode else "Single-label"}')

        start_time = time.time()
        if progress_callback:
            progress_callback('status', 'Splitting data (stratified)...')

        if test_split > 0:
            train_paths, val_paths, self.test_paths, train_labels, val_labels, self.test_labels = \
                self._three_way_split(image_paths, labels, validation_split, test_split)
            if progress_callback:
                progress_callback('detail', f'Test set (held-out): {len(self.test_paths)} images')
        else:
            self.test_paths, self.test_labels = [], []
            train_paths, val_paths, train_labels, val_labels = self._stratified_split(
                image_paths, labels, validation_split
            )

        if progress_callback:
            progress_callback('detail', f'Training set:   {len(train_paths)} images (stratified)')
            progress_callback('detail', f'Validation set: {len(val_paths)} images (stratified)')

        # ── Class counts / distribution logging ──────────────────────────
        if self.multi_label_mode:
            num_classes = len(labels[0])
            tag_counts  = np.array(train_labels).sum(axis=0)
            if progress_callback:
                progress_callback('detail', f'Number of tags: {num_classes}')
                top_idx = np.argsort(tag_counts)[::-1][:10]
                for idx in top_idx:
                    cnt = int(tag_counts[idx])
                    progress_callback('detail',
                        f'  {self.label_encoder.classes_[idx]}: {cnt} ({100*cnt/len(train_labels):.1f}%)')
                max_c = tag_counts.max()
                min_c = tag_counts[tag_counts > 0].min() if (tag_counts > 0).any() else 1
                ratio = max_c / min_c if min_c > 0 else 1.0
                progress_callback('detail', f'Tag imbalance ratio: {ratio:.2f}x')
                if ratio > 3.0:
                    progress_callback('detail', '⚠ Significant tag imbalance — using pos_weight balancing')
        else:
            class_counts = Counter(train_labels)
            num_classes  = len(self.label_encoder.classes_)
            if progress_callback:
                progress_callback('detail', f'Number of classes: {num_classes}')
                for ci, cnt in sorted(class_counts.items()):
                    progress_callback('detail',
                        f'  {self.label_encoder.classes_[ci]}: {cnt} ({100*cnt/len(train_labels):.1f}%)')

        # ── Datasets and DataLoaders ──────────────────────────────────────
        pre          = ImagePreprocessor()
        use_pin      = self.device.type == 'cuda'
        train_loader = DataLoader(
            ImageDataset(train_paths, train_labels, pre.train_transform, multi_label=self.multi_label_mode),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=_SAFE_WORKERS, pin_memory=use_pin,
            persistent_workers=(_SAFE_WORKERS > 0),
        )
        val_loader   = DataLoader(
            ImageDataset(val_paths, val_labels, pre.transform, multi_label=self.multi_label_mode),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=_SAFE_WORKERS, pin_memory=use_pin,
            persistent_workers=(_SAFE_WORKERS > 0),
        )

        # ── Model initialisation ──────────────────────────────────────────
        if progress_callback:
            progress_callback('status', 'Initializing model...')
            progress_callback('detail', f'Model architecture: {config_manager.app_config.model_type}')
        self.model = ImageClassificationModel(num_classes=num_classes, pretrained=True)
        self.model.to(self.device)

        # ── Loss function ─────────────────────────────────────────────────
        if self.multi_label_mode:
            tag_counts  = np.array(train_labels).sum(axis=0)
            n_train     = len(train_labels)
            pos_weights = torch.tensor(
                [(n_train - c) / c if c > 0 else 1.0 for c in tag_counts],
                dtype=torch.float32, device=self.device,
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            if progress_callback:
                progress_callback('detail', 'Loss: BCEWithLogitsLoss (pos_weight balanced)')
        else:
            class_counts_d = {}
            for lbl in train_labels:
                class_counts_d[lbl] = class_counts_d.get(lbl, 0) + 1
            cw = torch.tensor(
                [len(train_labels) / (num_classes * class_counts_d.get(i, 1))
                 for i in range(num_classes)],
                dtype=torch.float32, device=self.device,
            )
            criterion = nn.CrossEntropyLoss(weight=cw)
            if progress_callback:
                progress_callback('detail', 'Loss: CrossEntropyLoss (single-label)')

        optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        use_amp   = _AMP_AVAILABLE and self.device.type == 'cuda'
        scaler    = torch.amp.GradScaler('cuda') if use_amp else None  # type: ignore[attr-defined]

        if progress_callback:
            progress_callback('detail', f'LR: {cfg.learning_rate}  |  Batch: {cfg.batch_size}  |  AMP: {use_amp}')
            progress_callback('status', 'Starting training...')

        # ── Epoch loop ────────────────────────────────────────────────────
        best_accuracy    = 0.0
        patience_counter = 0
        train_accuracies: List[float] = []
        val_accuracies:   List[float] = []
        epoch_times:      List[float] = []
        total_batches = len(train_loader)
        cancelled = False

        for epoch in range(cfg.epochs):
            self.model.train()
            train_loss = train_correct = train_total = 0.0
            t_epoch = time.time()

            for batch_idx, (images, batch_labels) in enumerate(train_loader):
                images       = images.to(self.device, non_blocking=use_pin)
                batch_labels = batch_labels.to(self.device, non_blocking=use_pin)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):  # type: ignore[attr-defined]
                        outputs = self.model(images)
                        loss    = criterion(outputs, batch_labels)
                    scaler.scale(loss).backward()   # type: ignore[union-attr]
                    scaler.step(optimizer)           # type: ignore[union-attr]
                    scaler.update()                  # type: ignore[union-attr]
                else:
                    outputs = self.model(images)
                    loss    = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                if self.multi_label_mode:
                    preds          = torch.sigmoid(outputs) > 0.5
                    train_correct += (preds == batch_labels.bool()).float().sum().item()
                    train_total   += batch_labels.numel()
                else:
                    _, pred        = torch.max(outputs.data, 1)
                    train_total   += batch_labels.size(0)
                    train_correct += (pred == batch_labels).sum().item()

                if progress_callback and total_batches > 0 and (
                    batch_idx % 5 == 0 or batch_idx == total_batches - 1
                ):
                    pct = 30 + (epoch + (batch_idx + 1) / total_batches) / cfg.epochs * 70
                    progress_callback('progress', pct)
                    progress_callback('status',
                        f'Epoch {epoch+1}/{cfg.epochs} — batch {batch_idx+1}/{total_batches}')

            epoch_elapsed  = time.time() - t_epoch
            epoch_times.append(epoch_elapsed)
            train_acc      = 100 * train_correct / train_total if train_total > 0 else 0.0
            avg_loss       = train_loss / len(train_loader) if train_loader else 0.0
            val_acc, val_macro_f1, val_micro_f1 = self.validate(val_loader)

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            self.training_metrics['losses'].append(avg_loss)
            self.training_metrics['train_accuracies'] = train_accuracies
            self.training_metrics['val_accuracies']   = val_accuracies
            self.training_metrics['accuracies']       = val_accuracies
            self.training_metrics['epochs_completed'] = epoch + 1
            self.training_metrics['val_macro_f1'].append(val_macro_f1)
            self.training_metrics['val_micro_f1'].append(val_micro_f1)

            gap = train_acc - val_acc
            remaining   = cfg.epochs - (epoch + 1)
            avg_e       = sum(epoch_times) / len(epoch_times)
            eta_s       = avg_e * remaining
            eta_str     = (f"{int(eta_s//3600)}h {int(eta_s%3600//60)}m" if eta_s >= 3600
                           else f"{int(eta_s//60)}m {int(eta_s%60)}s" if eta_s >= 60
                           else f"{int(eta_s)}s")
            eta_label   = f"ETA: {eta_str}" if remaining > 0 else "Done"
            epoch_msg   = (f"Epoch {epoch+1}/{cfg.epochs}  Loss: {avg_loss:.4f}  "
                           f"Train: {train_acc:.2f}%  Val: {val_acc:.2f}%  "
                           f"MacroF1: {val_macro_f1:.4f}  MicroF1: {val_micro_f1:.4f}  {eta_label}")
            print(epoch_msg)

            if progress_callback:
                progress_callback('status',  f'Training: Epoch {epoch+1}/{cfg.epochs} — {eta_label}')
                progress_callback('detail',  epoch_msg)
                progress_callback('metrics', {
                    'train_acc': train_acc, 'val_acc': val_acc, 'overfitting_gap': gap,
                    'val_macro_f1': val_macro_f1, 'val_micro_f1': val_micro_f1,
                })
                progress_callback('progress', 30 + (epoch + 1) / cfg.epochs * 70)
                if epoch >= 2:
                    if gap > 15:
                        progress_callback('detail', f'⚠ Overfitting: train-val gap = {gap:.1f}%')
                    elif train_acc < 50 and val_acc < 50:
                        progress_callback('detail', '⚠ Possible underfitting: both accuracies low')

            if cancel_event and cancel_event.is_set():
                cancelled = True
                if progress_callback:
                    progress_callback('cancelled', True)
                break

            if val_acc > best_accuracy:
                best_accuracy    = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        elapsed = time.time() - start_time
        self.training_metrics['total_time'] = elapsed
        self.training_metrics['cancelled']  = cancelled

        if not cancelled:
            if progress_callback:
                progress_callback('status', 'Saving model...')
            self.save_model('latest')
            if progress_callback:
                progress_callback('detail', f'Training completed in {elapsed:.2f}s')
                progress_callback('detail', f'Best validation accuracy: {best_accuracy:.2f}%')

            # Evaluate on the held-out test set while the model is still in memory.
            if test_split > 0 and self.test_paths:
                self.evaluate_on_test_set(progress_callback=progress_callback)

        # Release the training model from memory — the inference engine loads its
        # own copy via _auto_load_model(), so keeping this around just wastes RAM/VRAM.
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.training_metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Returns (accuracy %, macro_f1, micro_f1)."""
        if self.model is None:
            return 0.0, 0.0, 0.0
        self.model.eval()
        correct = total = 0
        all_preds:  List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                if self.multi_label_mode:
                    preds   = (torch.sigmoid(outputs) > 0.5)
                    correct += (preds == labels.bool()).float().sum().item()
                    total   += labels.numel()
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.bool().cpu().numpy())
                else:
                    _, pred  = torch.max(outputs.data, 1)
                    total   += labels.size(0)
                    correct += (pred == labels).sum().item()
                    all_preds.append(pred.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

        acc = 100.0 * correct / total if total > 0 else 0.0

        macro_f1 = micro_f1 = 0.0
        if all_preds:
            from sklearn.metrics import f1_score
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_labels, axis=0)
            self._last_val_preds  = y_pred
            self._last_val_labels = y_true
            try:
                macro_f1 = float(f1_score(y_true, y_pred, average='macro',  zero_division=0))
                micro_f1 = float(f1_score(y_true, y_pred, average='micro',  zero_division=0))
            except Exception:
                pass

        return acc, macro_f1, micro_f1

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, model_name: str = 'latest') -> None:
        if self.model is None:
            print("No model to save")
            return
        model_type  = config_manager.app_config.model_type
        model_state = {'model_state': self.model.state_dict()}
        metadata    = {
            'num_classes':       len(self.label_encoder.classes_),
            'classes':           self.label_encoder.classes_.tolist(),
            'multi_label':       self.multi_label_mode,
            'training_metrics':  self.training_metrics,
            'hyperparameters':   config_manager.hyperparameters.to_dict(),
            'model_type':        model_type,
            'data_source':       self.data_source,
        }
        print("[DEBUG] About to save model weights...", flush=True)
        try:
            model_storage.save_model_weights(model_state, model_name, metadata)
            print("[DEBUG] Model weights saved.", flush=True)
        except Exception as exc:
            print(f"Error saving model weights: {exc}", flush=True)
            return

        print("[DEBUG] About to save training stats...", flush=True)
        try:
            model_storage.save_training_stats(self.training_metrics, model_type)
            print("[DEBUG] Training stats saved.", flush=True)
        except Exception as exc:
            print(f"Warning: could not save training stats: {exc}", flush=True)

        if _MATPLOTLIB_AVAILABLE:
            print("[DEBUG] About to save training graphs (matplotlib)...", flush=True)
            try:
                self._save_training_graphs(model_type)
                print("[DEBUG] Training graphs saved.", flush=True)
            except Exception as exc:
                print(f"Warning: could not save training graphs: {exc}", flush=True)
            print("[DEBUG] About to save confusion matrix...", flush=True)
            try:
                self._save_confusion_matrix(model_type)
                print("[DEBUG] Confusion matrix saved.", flush=True)
            except Exception as exc:
                print(f"Warning: could not save confusion matrix: {exc}", flush=True)

        final_acc = self.training_metrics['accuracies'][-1] if self.training_metrics['accuracies'] else 0
        print("[DEBUG] About to log training session to DB...", flush=True)
        try:
            metadata_store.log_training_session(
                model_name=model_name,
                epochs=self.training_metrics['epochs_completed'],
                final_accuracy=final_acc,
                hyperparameters=config_manager.hyperparameters.to_dict(),
                notes=f"Training completed in {self.training_metrics['total_time']:.2f}s "
                      f"({'multi-label' if self.multi_label_mode else 'single-label'})",
            )
            print("[DEBUG] Training session logged.", flush=True)
        except Exception as exc:
            print(f"Warning: could not log training session: {exc}", flush=True)

    def evaluate_on_test_set(
        self, progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run the trained model on the held-out test set and return full metrics.

        Must be called after train() with test_split > 0.
        Returns a dict with accuracy, macro_f1, micro_f1, and per-class F1.
        """
        if self.model is None:
            return {'error': 'No model available'}
        if not self.test_paths:
            return {'error': 'No held-out test set — re-run train() with test_split > 0'}

        if progress_callback:
            progress_callback('status', f'Evaluating on held-out test set ({len(self.test_paths)} images)…')

        pre  = ImagePreprocessor()
        use_pin = self.device.type == 'cuda'
        loader  = DataLoader(
            ImageDataset(self.test_paths, self.test_labels, pre.transform,
                         multi_label=self.multi_label_mode),
            batch_size=config_manager.hyperparameters.batch_size, shuffle=False,
            num_workers=_SAFE_WORKERS, pin_memory=use_pin,
            persistent_workers=(_SAFE_WORKERS > 0),
        )

        test_acc, macro_f1, micro_f1 = self.validate(loader)

        # Per-class F1
        per_class_f1: Dict[str, float] = {}
        self.model.eval()
        all_preds:  List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                if self.multi_label_mode:
                    preds = (torch.sigmoid(outputs) > 0.5)
                else:
                    _, preds = torch.max(outputs.data, 1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.bool().cpu().numpy() if self.multi_label_mode
                                  else labels.cpu().numpy())

        if all_preds:
            from sklearn.metrics import f1_score
            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_labels, axis=0)
            try:
                per_class_scores = np.asarray(
                    f1_score(y_true, y_pred, average=None, zero_division=0)
                )
                classes = self.label_encoder.classes_
                per_class_f1 = {
                    str(classes[i]): float(per_class_scores[i])
                    for i in range(len(per_class_scores))
                    if i < len(classes)
                }
            except Exception:
                pass

        result = {
            'test_accuracy':  test_acc,
            'macro_f1':       macro_f1,
            'micro_f1':       micro_f1,
            'per_class_f1':   per_class_f1,
            'n_test_images':  len(self.test_paths),
        }
        print(f"\n{'='*55}")
        print(f"TEST SET RESULTS  ({len(self.test_paths)} images)")
        print(f"  Accuracy:  {test_acc:.2f}%")
        print(f"  Macro F1:  {macro_f1:.4f}")
        print(f"  Micro F1:  {micro_f1:.4f}")
        if per_class_f1:
            print("  Per-class F1 (top 10 by score):")
            for cls, sc in sorted(per_class_f1.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                print(f"    {cls}: {sc:.4f}")
        print('='*55)

        if progress_callback:
            progress_callback('detail', f'Test Acc: {test_acc:.2f}%  MacroF1: {macro_f1:.4f}  MicroF1: {micro_f1:.4f}')

        self.training_metrics['test_results'] = result
        try:
            model_type = config_manager.app_config.model_type
            model_storage.save_test_results(result, model_type)
        except Exception as exc:
            print(f"Warning: could not save test results: {exc}", flush=True)
        return result

    def _save_training_graphs(self, model_type: str) -> None:
        epochs_done = self.training_metrics.get('epochs_completed', 0)
        if epochs_done == 0 or _plt is None:
            return
        x = list(range(1, epochs_done + 1))
        train_accs = self.training_metrics.get('train_accuracies', [])
        val_accs   = self.training_metrics.get('val_accuracies',   [])
        if train_accs or val_accs:
            fig, ax = _plt.subplots(figsize=(8, 5))
            if train_accs:
                ax.plot(x[:len(train_accs)], train_accs, label='Train Accuracy', marker='o', linewidth=2)
            if val_accs:
                ax.plot(x[:len(val_accs)],   val_accs,   label='Val Accuracy',   marker='s', linewidth=2)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Training Accuracy — {model_type}')
            ax.legend(); ax.grid(True, alpha=0.3)
            model_storage.save_training_graph(fig, 'accuracy', model_type)
            _plt.close(fig)
        losses = self.training_metrics.get('losses', [])
        if losses:
            fig, ax = _plt.subplots(figsize=(8, 5))
            ax.plot(x[:len(losses)], losses, label='Train Loss', color='red', marker='o', linewidth=2)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(f'Training Loss — {model_type}')
            ax.legend(); ax.grid(True, alpha=0.3)
            model_storage.save_training_graph(fig, 'loss', model_type)
            _plt.close(fig)

        macro_f1s = self.training_metrics.get('val_macro_f1', [])
        micro_f1s = self.training_metrics.get('val_micro_f1', [])
        if macro_f1s or micro_f1s:
            fig, ax = _plt.subplots(figsize=(8, 5))
            if macro_f1s:
                ax.plot(x[:len(macro_f1s)], macro_f1s, label='Val Macro F1', marker='^', linewidth=2)
            if micro_f1s:
                ax.plot(x[:len(micro_f1s)], micro_f1s, label='Val Micro F1', marker='v', linewidth=2, linestyle='--')
            ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score')
            ax.set_title(f'Validation F1 Score — {model_type}')
            ax.legend(); ax.grid(True, alpha=0.3)
            model_storage.save_training_graph(fig, 'f1', model_type)
            _plt.close(fig)

    # ------------------------------------------------------------------
    # Confusion matrix / per-class analysis
    # ------------------------------------------------------------------

    def _save_confusion_matrix(self, model_type: str) -> None:
        """Dispatch to single- or multi-label confusion figure based on mode."""
        if _plt is None:
            return
        if self._last_val_preds is None or self._last_val_labels is None:
            return
        classes = np.asarray(self.label_encoder.classes_)
        if len(classes) == 0:
            return
        if self.multi_label_mode:
            self._save_multilabel_confusion(
                self._last_val_preds, self._last_val_labels, classes, model_type)
        else:
            self._save_singlelabel_confusion(
                self._last_val_preds, self._last_val_labels, classes, model_type)

    def _save_multilabel_confusion(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        classes: np.ndarray,
        model_type: str,
    ) -> None:
        from sklearn.metrics import f1_score

        n_classes = len(classes)
        support   = y_true.sum(axis=0).astype(int)
        per_f1    = np.asarray(f1_score(y_true, y_pred, average=None, zero_division=0), dtype=np.float64)

        # ── Figure 1: Per-class F1 bar chart ─────────────────────────
        top_idx  = np.argsort(support)[::-1][:30]
        bar_idx  = top_idx[np.argsort(per_f1[top_idx])]   # ascending F1
        n_bars   = len(bar_idx)
        fig_h    = max(6, n_bars * 0.38)
        fig, ax  = _plt.subplots(figsize=(10, fig_h))
        colors   = [
            '#d9534f' if f < 0.2 else '#f0ad4e' if f < 0.5 else '#5cb85c'
            for f in per_f1[bar_idx]
        ]
        ax.barh(np.arange(n_bars), per_f1[bar_idx], color=colors, alpha=0.85)
        ax.set_yticks(np.arange(n_bars))
        ax.set_yticklabels(
            [f"{classes[i]}  (n={support[i]})" for i in bar_idx], fontsize=8)
        ax.set_xlabel('F1 Score')
        ax.set_xlim(0, 1.0)
        mean_f1 = float(per_f1.mean())
        ax.axvline(mean_f1, color='navy', linestyle='--', linewidth=1.2,
                   label=f'Mean F1: {mean_f1:.3f}')
        ax.set_title(
            f'Per-Class F1 — {model_type}  (top {n_bars} by support)')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        _plt.tight_layout()
        model_storage.save_training_graph(fig, 'per_class_f1', model_type)
        _plt.close(fig)

        # ── Figure 2: Co-confusion heatmap ────────────────────────────
        # C[i,j] = times truth class i was missed (FN) while class j was
        # predicted (FP), i.e. which wrong classes fired when we missed i.
        top_n   = min(15, n_classes)
        top_idx = np.argsort(support)[::-1][:top_n]
        top_cls = classes[top_idx]
        C = np.zeros((top_n, top_n), dtype=int)
        for i, ci in enumerate(top_idx):
            fn_mask = (y_true[:, ci] == 1) & (y_pred[:, ci] == 0)
            if not fn_mask.any():
                continue
            for j, cj in enumerate(top_idx):
                if i != j:
                    C[i, j] = int((fn_mask & (y_pred[:, cj] == 1)).sum())

        fig, ax = _plt.subplots(figsize=(10, 8))
        im = ax.imshow(C, cmap='YlOrRd', aspect='auto', vmin=0)
        _plt.colorbar(im, ax=ax, label='Count')
        ax.set_xticks(range(top_n))
        ax.set_yticks(range(top_n))
        ax.set_xticklabels(top_cls, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(top_cls, fontsize=8)
        ax.set_xlabel('Incorrectly Predicted Class (False Positive)')
        ax.set_ylabel('Missed True Class (False Negative)')
        ax.set_title(
            f'Co-Confusion Matrix — {model_type}\n'
            f'Row = missed truth  |  Col = wrong prediction')
        vmax = C.max() if C.max() > 0 else 1
        for i in range(top_n):
            for j in range(top_n):
                if C[i, j] > 0:
                    ax.text(j, i, str(C[i, j]), ha='center', va='center',
                            fontsize=7,
                            color='white' if C[i, j] > vmax * 0.6 else 'black')
        _plt.tight_layout()
        model_storage.save_training_graph(fig, 'confusion_matrix', model_type)
        _plt.close(fig)

    def _save_singlelabel_confusion(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        classes: np.ndarray,
        model_type: str,
    ) -> None:
        from sklearn.metrics import confusion_matrix, f1_score

        n_classes   = len(classes)
        max_display = 20
        if n_classes > max_display:
            counts  = np.bincount(y_true.astype(int), minlength=n_classes)
            top_idx = np.argsort(counts)[::-1][:max_display]
            mask    = np.isin(y_true, top_idx)
            y_t, y_p  = y_true[mask], y_pred[mask]
            disp_cls  = classes[top_idx]
            labels    = top_idx.tolist()
        else:
            y_t, y_p  = y_true, y_pred
            disp_cls  = classes
            labels    = list(range(n_classes))

        cm      = confusion_matrix(y_t, y_p, labels=labels)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, where=row_sum != 0)

        n_d    = len(disp_cls)
        fs     = max(6, 11 - n_d // 5)
        fig_sz = max(8, n_d * 0.55)
        fig, ax = _plt.subplots(figsize=(fig_sz, fig_sz))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        _plt.colorbar(im, ax=ax, label='Recall (row-normalised)')
        ax.set_xticks(range(n_d))
        ax.set_yticks(range(n_d))
        ax.set_xticklabels(disp_cls, rotation=45, ha='right', fontsize=fs)
        ax.set_yticklabels(disp_cls, fontsize=fs)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        title = f'Confusion Matrix — {model_type}'
        if n_classes > max_display:
            title += f' (top {max_display} classes by frequency)'
        ax.set_title(title)
        for i in range(n_d):
            ax.text(i, i, f'{cm_norm[i, i]:.2f}', ha='center', va='center',
                    fontsize=max(5, fs - 1),
                    color='white' if cm_norm[i, i] > 0.6 else 'black')
        _plt.tight_layout()
        model_storage.save_training_graph(fig, 'confusion_matrix', model_type)
        _plt.close(fig)

        # Per-class F1 bar chart
        per_f1  = np.asarray(f1_score(y_true, y_pred, average=None, zero_division=0,
                                       labels=list(range(n_classes))), dtype=np.float64)
        sort_i  = np.argsort(per_f1)
        fig_h   = max(6, n_classes * 0.38)
        fig, ax = _plt.subplots(figsize=(10, fig_h))
        colors  = [
            '#d9534f' if f < 0.2 else '#f0ad4e' if f < 0.5 else '#5cb85c'
            for f in per_f1[sort_i]
        ]
        ax.barh(np.arange(n_classes), per_f1[sort_i], color=colors, alpha=0.85)
        ax.set_yticks(np.arange(n_classes))
        ax.set_yticklabels([str(classes[i]) for i in sort_i], fontsize=8)
        ax.set_xlabel('F1 Score')
        ax.set_xlim(0, 1.0)
        mean_f1 = float(per_f1.mean())
        ax.axvline(mean_f1, color='navy', linestyle='--', linewidth=1.2,
                   label=f'Mean F1: {mean_f1:.3f}')
        ax.set_title(f'Per-Class F1 — {model_type}')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        _plt.tight_layout()
        model_storage.save_training_graph(fig, 'per_class_f1', model_type)
        _plt.close(fig)

    # ------------------------------------------------------------------
    # Inbox fine-tuning
    # ------------------------------------------------------------------

    def fine_tune_from_annotations(
        self,
        annotations: Dict[str, List[str]],
        epochs: int = 5,
        lr: float = 0.0001,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Fine-tune the current model on a small set of user-labeled inbox images."""
        if self.model is None:
            if progress_callback:
                progress_callback('status', 'Loading existing model weights…')
            try:
                model_state, meta         = model_storage.load_model_weights('latest')
                num_classes               = meta.get('num_classes', 10)
                self.multi_label_mode     = meta.get('multi_label', True)
                self.model                = ImageClassificationModel(num_classes=num_classes)
                self.model.load_state_dict(model_state['model_state'])
                self.model.to(self.device)
                self.label_encoder.classes_ = np.array(meta.get('classes', []))
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

        self.multi_label_mode = True
        tag_to_idx  = {t: i for i, t in enumerate(known_classes)}
        n_classes   = len(known_classes)
        image_paths: List[str]         = []
        multi_labels: List[List[float]] = []

        for img_path, tags in annotations.items():
            if not Path(img_path).exists():
                continue
            vec     = [0.0] * n_classes
            matched = False
            for tag in tags:
                if tag in tag_to_idx:
                    vec[tag_to_idx[tag]] = 1.0
                    matched              = True
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
            progress_callback('status', f'Fine-tuning on {n} inbox image(s) — {epochs} epoch(s)…')
            progress_callback('detail',  f'Classes: {n_classes}  |  LR: {lr}  |  Epochs: {epochs}')

        pre       = ImagePreprocessor()
        dataset   = ImageDataset(image_paths, multi_labels, pre.transform, multi_label=True)
        loader    = DataLoader(dataset, batch_size=min(8, n), shuffle=True, num_workers=0)

        # ── Baseline validation (before fine-tuning) ──────────────────────
        self.model.eval()
        val_loader_fixed = DataLoader(
            dataset, batch_size=min(8, n), shuffle=False, num_workers=0
        )
        pre_acc, pre_macro_f1, pre_micro_f1 = self.validate(val_loader_fixed)
        print(f"[Fine-tune] PRE-update  → Acc: {pre_acc:.2f}%  MacroF1: {pre_macro_f1:.4f}  MicroF1: {pre_micro_f1:.4f}")
        if progress_callback:
            progress_callback('detail', f'PRE fine-tune  → Acc: {pre_acc:.2f}%  MacroF1: {pre_macro_f1:.4f}')

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        t0        = time.time()
        ft_losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, lbls in loader:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(imgs), lbls)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(len(loader), 1)
            ft_losses.append(avg_loss)
            msg = f'Fine-tune epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}'
            print(msg)
            if progress_callback:
                progress_callback('status',   msg)
                progress_callback('progress', 100 * (epoch + 1) / epochs)

        elapsed = time.time() - t0

        # ── Post-update validation ────────────────────────────────────────
        self.model.eval()
        post_acc, post_macro_f1, post_micro_f1 = self.validate(val_loader_fixed)
        print(f"[Fine-tune] POST-update → Acc: {post_acc:.2f}%  MacroF1: {post_macro_f1:.4f}  MicroF1: {post_micro_f1:.4f}")
        if progress_callback:
            progress_callback('detail', f'POST fine-tune → Acc: {post_acc:.2f}%  MacroF1: {post_macro_f1:.4f}')

        if progress_callback:
            progress_callback('status', 'Saving fine-tuned model…')
        self.save_model('latest')
        ft_metrics: Dict[str, Any] = {
            'losses':           ft_losses,
            'accuracies':       [],
            'epochs_completed': epochs,
            'total_time':       elapsed,
            'n_images':         n,
            'fine_tune':        True,
            # Before / after validation deltas
            'pre_accuracy':     pre_acc,
            'pre_macro_f1':     pre_macro_f1,
            'pre_micro_f1':     pre_micro_f1,
            'post_accuracy':    post_acc,
            'post_macro_f1':    post_macro_f1,
            'post_micro_f1':    post_micro_f1,
            'delta_accuracy':   post_acc    - pre_acc,
            'delta_macro_f1':   post_macro_f1 - pre_macro_f1,
        }
        if progress_callback:
            progress_callback('detail',
                f'Fine-tuning done in {elapsed:.1f}s  |  Final loss: {ft_losses[-1]:.4f}  |  '
                f'ΔMacroF1: {ft_metrics["delta_macro_f1"]:+.4f}')
            progress_callback('complete', True)

        # ── Persist fine-tune metrics to disk ─────────────────────────────
        try:
            model_type = config_manager.app_config.model_type
            model_storage.save_training_stats(ft_metrics, model_type)
            print(f"[Fine-tune] Metrics saved to data_training_graphs/")
        except Exception as exc:
            print(f"Warning: could not save fine-tune metrics: {exc}", flush=True)

        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ft_metrics
