# Section 3 — Methodology: Ready-to-Insert Text Blocks

All blocks below are ready to copy directly into the article.
Replace bracketed placeholders `[X]` with values from your training run output.

---

## 3.1  Revised Opening Sentence  (Item 6)

**Remove:**
> "For the application, greater focus will be placed on the automatic sorting systems rather
> than the file explorer, prioritizing the automated sorting mechanisms, as the file explorer
> serves primarily as the functional framework for demonstrating the user-oriented training
> process."

**Replace with:**
> "This chapter focuses primarily on the automatic sorting systems, as the file explorer
> serves mainly as the interface through which the user-oriented training process is
> demonstrated."

---

## 3.2  Dataset Partitioning Paragraph  (Items 3 & 5)

Insert at the **end of the dataset description sub-section** in Section 3.

> "Prior to training, the dataset was partitioned into three non-overlapping subsets using
> stratified sampling to preserve class distribution across all splits: a training set
> (70 %, [X_train] images), a validation set (15 %, [X_val] images), and a held-out test set
> (15 %, [X_test] images).  The test set was withheld entirely during all training and
> hyperparameter decisions; final generalisation performance was evaluated on it only once,
> after the best model had been selected on the validation set.  Given the severe class
> imbalance across the [N_tags] tag categories, stratified sampling was considered essential
> to ensure that minority classes appeared in all three partitions."

**How to fill placeholders:**
After a training run, open the training progress dialog or terminal output and look for lines such as:

```
Training set:   [X_train] images (stratified)
Validation set: [X_val] images (stratified)
Test set (held-out): [X_test] images
```

These are emitted by `training.py` at the start of every `train()` call.

---

## 3.3  Transfer Learning Justification  (Item 7)

Insert after the model architecture description paragraph.

> "Transfer learning from ImageNet-pretrained weights is particularly well-suited to this
> task for two reasons.  First, personal photo collections occupy a visual domain that
> substantially overlaps with ImageNet categories; consequently, low-level convolutional
> filters trained on ImageNet (edges, textures, object parts) transfer directly without
> domain-shift penalties.  Second, the target dataset is too small to train a deep network
> from random initialisation without severe overfitting, a constraint well-documented in the
> few-shot and transfer learning literature [CITE_TRANSFER_1][CITE_TRANSFER_2]."

---

## 3.4  Privacy / Local-Execution Properties  (Item 1 — alternative to title change)

If you prefer to retain the word "Secure" in the title rather than changing it, add this
paragraph to Section 3 (system description sub-section).

> "The system provides the following privacy and data-sovereignty guarantees by design:
> (i) all inference runs locally; no image data or model weights are transmitted to external
> services;
> (ii) the training pipeline operates entirely on the user's own hardware and file-system,
> with no dependency on cloud compute or remote APIs;
> (iii) COCO benchmark images are the only data retrieved over the network, and solely for
> the purpose of reproducible evaluation — personal photo collections never leave the
> user's machine."

**Verification checklist against the implementation:**
- [x] Inference path: `InferenceEngine.predict_single()` / `predict_batch()` in `ml/inference.py` — pure local PyTorch, no HTTP calls.
- [x] Training path: `TrainingLoop.train()` in `ml/training.py` — no network calls.
- [x] Model persistence: `ModelStorage.save_model_weights()` in `data_layer.py` — writes only to local filesystem paths under `ML_Training_Data/model_folder/`.
- [ ] COCO download: `COCOBenchmarkLoader._ensure_image()` in `coco_benchmark.py` **does** make HTTP calls to `images.cocodataset.org` — this should be explicitly acknowledged in the text as an opt-in benchmark-only operation, not part of normal usage.
