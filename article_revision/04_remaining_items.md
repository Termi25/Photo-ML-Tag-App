# Remaining Items Checklist
# (things that require you to act — code is done, data is not yet generated)

---

## CRITICAL — Must complete before re-submission

### A.  Run ResNet-18 full experiment (produces all primary data)

1. Open the app.
2. Select profile **"Paper — ResNet-18"** (or your existing hyperparameters).
3. Train using **"Train from Folders"** on your personal 2165-image dataset.
4. After training, terminal output contains:
   - All epoch rows for Table 2 (Train Acc, Val Acc, Loss, Macro F1, Micro F1).
   - The `TEST SET RESULTS` block for Table 2 final row and Table 3 (ResNet-18 row).
5. `ML_Training_Data/data_training_graphs/` will contain:
   - `graph_accuracy_<ts>.png`        → Fig. X (accuracy curves)
   - `graph_loss_<ts>.png`            → Fig. X (loss curve)
   - `graph_f1_<ts>.png`              → Fig. X (macro/micro F1 curves)
   - `graph_confusion_matrix_<ts>.png`→ Fig. X (co-confusion heatmap for Table 4)
   - `graph_per_class_f1_<ts>.png`    → Fig. X (per-class F1 bar chart)
   - `test_results_resnet18_<ts>.json`→ machine-readable data for all test metrics
   - `training_resnet18_<ts>.json`    → per-epoch metrics for Table 2

### B.  Run EfficientNet-B0 experiment (produces Table 3 second row)

1. Select profile **"Paper — EfficientNet-B0"**.
2. Train from the same folders with the **same random seed** (seed is fixed at 42 in code).
3. Collect the `TEST SET RESULTS` block from terminal for Table 3.
4. Collect `graph_confusion_matrix_<ts>.png` for a second confusion matrix figure
   (or compare directly with ResNet-18 matrix).

### C.  Fill Table 2 — F1 columns

From the terminal run in step A, copy `MacroF1` and `MicroF1` values into the template
in `02_table_templates.md`.

### D.  Fill Table 3 — EfficientNet row

From the terminal run in step B, copy `TEST SET RESULTS` into the template.

### E.  Fill Table 4 — Top misclassified pairs

1. Open `graph_confusion_matrix_<ts>.png` (from either run).
2. Identify the 5–10 darkest off-diagonal cells.
3. Each cell `(row i, col j)` = true class `i` was missed when class `j` was predicted.
4. Fill Table 4 in `02_table_templates.md`.

### F.  Run COCO experiment (for public-dataset reproducibility — Item 3)

1. Select profile **"COCO — ResNet-18"**.
2. Train using **"COCO only"** source.
3. This produces a separate set of metrics on 80 standard categories.
4. Add a paragraph or supplemental table reporting COCO test-set results to demonstrate
   reproducibility on a publicly licensed benchmark.

---

## HIGH — Article text (no experiments needed, ready to write)

- [ ] **Item 1** — Decide: change title OR add the privacy-properties paragraph from
      `01_methodology_section3.md` section 3.4.  Either is acceptable; the paragraph is
      ready to paste.
- [ ] **Item 5** — Paste the dataset-partitioning paragraph from `01_methodology_section3.md`
      section 3.2 and fill `[X_train]`, `[X_val]`, `[X_test]` from run A above.
- [ ] **Item 6** — Replace the opening sentence of Section 3 with the version in
      `01_methodology_section3.md` section 3.1.
- [ ] **Item 7 — justification text** — Paste the transfer learning paragraph from
      `01_methodology_section3.md` section 3.3.  Add the two citation keys.

---

## MEDIUM — Figures to include

| Figure | Source file | Where in article |
|--------|-------------|-----------------|
| Accuracy curves (train + val) | `graph_accuracy_<ts>.png` | Section 4, after Table 2 |
| Loss curve | `graph_loss_<ts>.png` | Section 4 |
| Macro/Micro F1 curves | `graph_f1_<ts>.png` | Section 4 |
| Co-confusion heatmap | `graph_confusion_matrix_<ts>.png` | Section 4, before Table 4 |
| Per-class F1 bar | `graph_per_class_f1_<ts>.png` | Appendix or Section 4 |

All are generated automatically — you just need to run the training (step A above).

---

## LOW — Formatting (Item 8)

- [ ] Open the submission file in the conference template.
- [ ] Verify the `\begin{abstract}...\end{abstract}` block is present and not commented out.
- [ ] Verify keywords appear under the correct style tag required by the template
      (e.g. `\keywords{...}` or a dedicated environment).
- No content change needed.

---

## NOT NEEDED — already done by code changes

- [x] Test split 70/15/15 implemented (`test_split=0.15` in all train callers).
- [x] `evaluate_on_test_set()` called automatically after every training run.
- [x] Test results saved to `test_results_<model>_<ts>.json`.
- [x] Macro F1 and Micro F1 logged per epoch (already existed; verified in code).
- [x] Confusion matrix auto-generated after training (already existed; verified in code).
- [x] Four new COCO/COCO+Folders profiles added to the profile selector.
- [x] Detail messages added to COCO loading so progress dialog shows activity.
