# Code Snippets for Article  (Section 3 — Implementation)

These snippets can be cited directly in the paper to substantiate implementation claims.
All line references point to the codebase as it stands after revision.

---

## Snippet 1 — Stratified Three-Way Split  (Items 3 & 5)

Demonstrates that the 70 / 15 / 15 split is implemented with stratification for both
single-label and multi-label label distributions.

**File:** `MyOwnApp/ml/training.py`

```python
def _three_way_split(self, image_paths, labels, val_split, test_split):
    """Stratified 70/15/15 split into train, val, and held-out test sets."""
    # First carve out the test set
    train_val_paths, test_paths, train_val_labels, test_labels = \
        self._stratified_split(image_paths, labels, test_split)
    # Split the remainder into train and val
    adjusted_val = val_split / (1.0 - test_split)
    train_paths, val_paths, train_labels, val_labels = \
        self._stratified_split(train_val_paths, train_val_labels, adjusted_val)
    return train_paths, val_paths, test_paths, \
           train_labels, val_labels, test_labels
```

In multi-label mode the iterative greedy assignment ensures that every tag category is
represented in all three partitions proportionally to its frequency:

```python
# Multi-label stratification (excerpt from _stratified_split)
tag_counts   = label_matrix.sum(axis=0)          # per-class image counts
desired_val  = tag_counts * validation_split      # target val count per class
for idx in rng.permutation(n):
    active  = np.where(label_matrix[idx] == 1)[0]
    deficit = desired_val[active] - val_tag_counts[active]
    if len(val_idx) < val_size and deficit.max() > 0:
        val_idx.append(idx)
        val_tag_counts[active] += 1
    else:
        train_idx.append(idx)
```

---

## Snippet 2 — Multi-Label Loss with Positive-Weight Balancing  (Item 2)

The BCEWithLogitsLoss is weighted by the inverse class frequency to counteract the severe
tag imbalance identified in the dataset, directly addressing the reviewer concern about
majority-class exploitation.

**File:** `MyOwnApp/ml/training.py`

```python
# Compute per-class positive weights from the training split
tag_counts  = np.array(train_labels).sum(axis=0)   # images per tag
n_train     = len(train_labels)
pos_weights = torch.tensor(
    [(n_train - c) / c if c > 0 else 1.0 for c in tag_counts],
    dtype=torch.float32, device=self.device,
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
```

A weight of `(N − c) / c` up-scales the loss contribution of rare tags and down-scales
that of frequent tags, which prevents the model from converging to a trivial majority-class
predictor.

---

## Snippet 3 — Per-Epoch F1 Computation  (Item 2)

F1 scores are computed at the end of every validation pass alongside accuracy.

**File:** `MyOwnApp/ml/training.py`

```python
def validate(self, val_loader):
    """Returns (accuracy %, macro_f1, micro_f1)."""
    ...
    from sklearn.metrics import f1_score
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    return acc, macro_f1, micro_f1
```

---

## Snippet 4 — Held-Out Test Set Evaluation  (Item 3)

Called once, after the final model is saved, before the model weights are released from
memory.

**File:** `MyOwnApp/ml/training.py`

```python
# Inside train(), after save_model():
if test_split > 0 and self.test_paths:
    self.evaluate_on_test_set(progress_callback=progress_callback)
```

```python
def evaluate_on_test_set(self, progress_callback=None):
    """Run the trained model on the held-out test set (one pass, never used for tuning)."""
    test_acc, macro_f1, micro_f1 = self.validate(loader)
    per_class_f1 = {
        str(classes[i]): float(per_class_scores[i])
        for i in range(len(per_class_scores))
    }
    result = {
        'test_accuracy': test_acc,
        'macro_f1':      macro_f1,
        'micro_f1':      micro_f1,
        'per_class_f1':  per_class_f1,
        'n_test_images': len(self.test_paths),
    }
    model_storage.save_test_results(result, model_type)
    return result
```

---

## Snippet 5 — Transfer Learning Architecture  (Item 7)

**File:** `MyOwnApp/ml/model.py`  (show this to substantiate the transfer learning claim)

Look for the `ImageClassificationModel` class and cite the `pretrained=True` constructor
parameter and the backbone freeze / unfreeze strategy if present.  The key claim is that
ImageNet weights are loaded and the classification head is replaced to match the number of
local tags.

---

## Snippet 6 — Confusion Matrix Generation  (Item 4)

Called automatically at the end of every training run.

**File:** `MyOwnApp/ml/training.py`

```python
# Inside save_model():
if _MATPLOTLIB_AVAILABLE:
    self._save_training_graphs(model_type)   # accuracy / loss / F1 curves
    self._save_confusion_matrix(model_type)  # per-class F1 bar + co-confusion heatmap
```

For multi-label models the co-confusion heatmap `C[i, j]` counts how many times
true class `i` was missed (false negative) while class `j` was spuriously predicted
(false positive):

```python
for i, ci in enumerate(top_idx):
    fn_mask = (y_true[:, ci] == 1) & (y_pred[:, ci] == 0)
    for j, cj in enumerate(top_idx):
        if i != j:
            C[i, j] = int((fn_mask & (y_pred[:, cj] == 1)).sum())
```

All graphs are saved to `ML_Training_Data/data_training_graphs/` and can be included
directly as figures in the article.
