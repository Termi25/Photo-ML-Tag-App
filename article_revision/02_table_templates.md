# Article Table Templates  (Items 2, 4, 7)

Fill every `[?]` cell from terminal output or the JSON files saved in
`ML_Training_Data/data_training_graphs/` after each run.

---

## Table 2 — Per-Epoch Training Metrics (ResNet-18, updated)

Replace the existing Table 2.  Macro F1 and Micro F1 columns are new.

| Epoch | Train Acc. | Val Acc. | Train Loss | Overfit Gap | Macro F1 | Micro F1 |
|-------|-----------|---------|------------|-------------|----------|----------|
| 1     | 63.19 %   | 67.22 % | 1.3521     | −4.03 pp    | [?]      | [?]      |
| 2     | 74.38 %   | 74.85 % | 1.3453     | −0.47 pp    | [?]      | [?]      |
| 3     | [?]       | [?]     | [?]        | [?]         | [?]      | [?]      |
| 4     | [?]       | [?]     | [?]        | [?]         | [?]      | [?]      |
| 5 (peak) | 66.96 % | 75.26 % | 1.3064  | −8.30 pp    | [?]      | [?]      |
| 6     | [?]       | [?]     | [?]        | [?]         | [?]      | [?]      |
| 7     | [?]       | [?]     | [?]        | [?]         | [?]      | [?]      |
| 8 (final) | 64.89 % | 54.52 % | 1.2639 | +10.37 pp   | [?]      | [?]      |

**Where to find the values:**
Every epoch line printed to the terminal (and shown in the training dialog) has the form:

```
Epoch N/8  Loss: X.XXXX  Train: XX.XX%  Val: XX.XX%  MacroF1: 0.XXXX  MicroF1: 0.XXXX  ETA: Xs
```

Copy those values directly.  The full run is also persisted as
`ML_Training_Data/data_training_graphs/training_resnet18_<timestamp>.json`.

**Caption (suggested):**
> Table 2.  Per-epoch training metrics for ResNet-18 fine-tuned on the personal photo
> collection (70 % train / 15 % val / 15 % test stratified split, multi-label
> BCEWithLogitsLoss).  Macro F1 reflects per-class average performance; a value
> substantially below validation accuracy indicates that the model exploits majority-class
> frequency rather than learning rare tags.

---

## Table 3 — Model Comparison  (Item 7)

New table.  Run the EfficientNet-B0 experiment using the "COCO — EfficientNet-B0" or
"Paper — EfficientNet-B0" profile, then fill the second row.

| Model           | Params (M) | Peak Val Acc. | Final Val Acc. | Test Acc. | Macro F1 (test) | Avg. Epoch Time (s) |
|-----------------|-----------|--------------|---------------|----------|-----------------|---------------------|
| ResNet-18       | 11.7      | 75.26 %      | 54.52 %       | [?]      | [?]             | 145.1               |
| EfficientNet-B0 | 5.3       | [?]          | [?]           | [?]      | [?]             | [?]                 |

**Where to find Test Acc. and Macro F1 (test):**
After training, the terminal prints a block:

```
=======================================================
TEST SET RESULTS  (N images)
  Accuracy:  XX.XX%
  Macro F1:  0.XXXX
  Micro F1:  0.XXXX
  Per-class F1 (top 10 by score): ...
=======================================================
```

The same data is saved to
`ML_Training_Data/data_training_graphs/test_results_<model>_<timestamp>.json`.

**Caption (suggested):**
> Table 3.  Comparison of ResNet-18 and EfficientNet-B0 on the held-out test set.
> EfficientNet-B0 achieves comparable accuracy with 54 % fewer parameters, supporting its
> selection for resource-constrained deployment.  All metrics are reported on the test set,
> which was withheld during all training and model-selection decisions.

---

## Table 4 — Top Misclassified Tag Pairs  (Item 4)

New table.  Values come from the confusion matrix saved to
`ML_Training_Data/data_training_graphs/graph_confusion_matrix_<timestamp>.png`
and from `test_results_<model>_<timestamp>.json` (the `per_class_f1` field).

| Rank | True Tag (ground truth) | Predicted Tag (false positive) | Co-confusion Count |
|------|------------------------|--------------------------------|--------------------|
| 1    | [?]                    | [?]                            | [?]                |
| 2    | [?]                    | [?]                            | [?]                |
| 3    | [?]                    | [?]                            | [?]                |
| 4    | [?]                    | [?]                            | [?]                |
| 5    | [?]                    | [?]                            | [?]                |

**How to extract top misclassified pairs:**
Open the saved `graph_confusion_matrix_<timestamp>.png` heatmap.  The colour intensity of
off-diagonal cells encodes co-confusion count.  The top-10 heaviest off-diagonal cells are
the pairs to list.  For the multi-label model the app generates a co-confusion heatmap
(rows = missed true class, columns = incorrectly predicted class) — read directly from it.

**Caption (suggested):**
> Table 4.  Top misclassified tag pairs ranked by co-confusion frequency.  A predicted tag
> is counted when the true label was missed (false negative on the row class) while the
> column class was simultaneously predicted (false positive).  Parent-child hierarchical
> pairs dominate, consistent with the label dependency discussed in Section 4.
