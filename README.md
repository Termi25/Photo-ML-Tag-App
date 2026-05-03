# Photo Categorization App — ML-Enhanced Image Sorting

A desktop image management application that uses machine learning to automatically categorize, tag, and sort large image collections. Built as part of doctoral research into ML applications for personal media organization.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Auto-Sorting Images](#auto-sorting-images)
  - [Tag Management](#tag-management)
  - [Searching by Tags](#searching-by-tags)
  - [Unsupervised Clustering](#unsupervised-clustering)
  - [Viewing Metrics](#viewing-metrics)
- [Configuration](#configuration)
- [ML Models Supported](#ml-models-supported)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Research Context](#research-context)
- [License](#license)

---

## Overview

This application provides a Windows-style file explorer interface on top of a machine learning pipeline, allowing users to:

- Browse image collections in detailed list or gallery view
- Manually tag images and train a classifier from those tags
- Bootstrap a model directly from the existing folder structure
- Automatically predict and apply tags to new, unsorted images
- Cluster images without any labels using K-means unsupervised learning
- Monitor prediction latency, training convergence, and sorting accuracy in real time

---

## Architecture

The application follows a clean **4-tier architecture**:

```
┌────────────────────────────────────────────────────┐
│           TIER 1: User Interface (UI)              │
│   file_explorer.py                                 │
│   - File Explorer (Detailed / Gallery views)       │
│   - Model Configuration Menu                       │
│   - Tag Management Sidebar                         │
└───────────────────────┬────────────────────────────┘
                        ↕
┌────────────────────────────────────────────────────┐
│         TIER 2: Logic Controller                   │
│   logic_controller.py                              │
│   - File Watcher (watchdog)                        │
│   - Tagging Engine                                 │
│   - Training Manager                               │
└───────────────────────┬────────────────────────────┘
                        ↕
┌────────────────────────────────────────────────────┐
│         TIER 3: ML Module                          │
│   ml_module.py                                     │
│   - Image Preprocessor                             │
│   - Inference Engine (7 backbone architectures)    │
│   - Training Loop (supervised + fine-tuning)       │
│   - Unsupervised Clustering (K-means)              │
└───────────────────────┬────────────────────────────┘
                        ↕
┌────────────────────────────────────────────────────┐
│         TIER 4: Data Layer                         │
│   data_layer.py                                    │
│   - SQLite Metadata Store                          │
│   - Model Weights Storage                          │
│   - Metrics & Training Graph Persistence           │
└────────────────────────────────────────────────────┘
```

---

## Features

### Core
| Feature | Description |
|---|---|
| File Explorer | Windows-style UI with detailed and gallery (thumbnail grid) views |
| Image Browsing | Navigate drives and directories with instant thumbnail rendering |
| Tag Management | Add, remove, and inspect tags with confidence scores and source info |
| Tag Search Bar | Persistent search bar in the main view; Boolean AND/OR search across the entire tagged image collection with live autocomplete |
| Multi-Tag Search | Boolean AND/OR search across tagged image collections |

### Machine Learning
| Feature | Description |
|---|---|
| Supervised Training | Train from manual tags or auto-bootstrap from folder hierarchy |
| Auto-Tagging | Predict and apply tags to new images using a trained model |
| Fine-Tuning | Continuously improve the model as the user corrects predictions |
| Unsupervised Clustering | Group visually similar images via K-means without any labels |
| Multiple Backbones | Switch between ResNet-18 and EfficientNet-B0 in configuration |

### Measurement & Research
| Metric | Description |
|---|---|
| Inference Latency | Per-image prediction time tracked in milliseconds |
| Training Convergence | Loss/accuracy curves saved as JSON and PNG graphs |
| Sorting Accuracy | Precision, Recall, and F1 against user-verified ground truth |
| Model Registry | Full metadata (architecture, date, classes) stored per model |

---

## Project Structure

```
App Project/
├── README.md                        ← You are here
├── ExampleImageFolder/              ← Sample image library (not committed)
│   ├── _INBOX/
│   ├── ANIME/
│   ├── ARCHITECTURE/
│   ├── ML_Training_Data/            ← Training graphs and model metadata
│   └── VIDEO GAMES/
│
├── MyOwnApp/                        ← Main application source
│   ├── file_explorer.py             # Entry point — Tkinter UI
│   ├── logic_controller.py          # Orchestration and file watching
│   ├── ml_module.py                 # ML pipeline (training + inference)
│   ├── data_layer.py                # SQLite store and model storage
│   ├── metrics.py                   # Performance tracking utilities
│   ├── config.py                    # Configuration management
│   ├── config.json                  # User-editable settings (auto-created)
│   ├── requirements.txt             # Python dependencies
│   └── data/
│       ├── image_metadata.db        # SQLite database (auto-created)
│       ├── models/                  # Saved model weights + metadata
│       ├── training_graphs/         # Loss/accuracy plots
│       └── cache/                   # Thumbnail cache
│
└── TagStudio-main/                  ← Reference implementation (analysis only)
```

---

## Getting Started

### Prerequisites

- Python **3.8+**
- pip
- *(Optional)* CUDA-capable GPU for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/photo-categ-app.git
   cd "photo-categ-app/App Project/MyOwnApp"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU support** *(optional but recommended)*
   ```bash
   # CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Running the Application

```bash
python file_explorer.py
```

The application opens to `../ExampleImageFolder` by default. Change `default_image_folder` in `config.json` to point to your own collection.

---

## Usage

### Quick Start Workflow

1. **Initialize**: Run the application via `python file_explorer.py`. It opens to the default directory (`../ExampleImageFolder`).
2. **Organize & Tag**: Browse your images, select ones you want to categorize, and add tags via the **Tag Management** sidebar.
3. **Train the Model**: Once you have enough tagged images (at least 2 categories), go to **Machine Learning → Train from Tags**. Alternatively, if your images are already neatly sorted in folders, use **Train from Folders**.
4. **Auto-Sort**: Navigate to a folder with untagged images, click **Machine Learning → Auto-Sort Images**, and apply the model's predicted tags.
5. **Search**: Use the **Tag Search** bar to quickly find images matching specific tags across your entire collection.

### Training a Model

**From folder structure** *(recommended for pre-organized collections)*
1. Navigate to a folder whose subfolders represent categories.
2. Go to **Machine Learning → Train from Folders**.
3. Training progress is shown in the status bar. The model is saved automatically.

**From manual tags**
1. Select an image → add tags in the **Tag Management** sidebar.
2. After tagging several images, go to **Machine Learning → Train from Tags**.

### Searching by Tags

A **Tag Search** bar is displayed below the navigation toolbar, always visible in the main window.

1. **Type** one or more tag names in the search field.
   - Separate multiple tags with **commas** (e.g. `HADES, VIDEO GAMES`).
   - An **autocomplete popup** appears as you type each token, listing matching tag names from the database. Double-click (or press `Enter`) on a suggestion to insert it.
2. **Choose a match mode** using the radio buttons to the right of the field:
   - **AND** — results must contain *all* specified tags (default).
   - **OR** — results contain *at least one* of the specified tags.
3. Press **Search** or hit `Enter` to execute the search.
   - Matching images are displayed in the current view (Detailed list or Gallery).
   - The status bar and the label next to the search field show how many images were found.
4. Press **✕ Clear** (or navigate to any folder) to exit search mode and return to the normal directory view.

> **Note:** The search operates across the entire database — it is not limited to the currently open folder. Images whose files no longer exist on disk are silently skipped in the results.

---

### Auto-Sorting Images

1. Navigate to a directory containing unsorted images.
2. Click **Machine Learning → Auto-Sort Images**.
3. Predicted tags with confidence scores appear in the sidebar.
4. Accept a suggestion with **Apply Selected Prediction** or add/remove tags manually.

### Tag Management

- **Add tag**: type in the input field → **Add Manual Tag** (marks as ground truth)
- **Remove tag**: select it in the list → **Remove Tag**
- **Confidence display**: each tag shows its source (`manual`, `model`, `cluster`) and score

### Unsupervised Clustering

1. Click **Machine Learning → Unsupervised Clustering**.
2. Enter the desired number of clusters.
3. Tags `cluster_0`, `cluster_1`, … are applied automatically.
4. Rename or merge clusters using the Tag Management sidebar.

### Viewing Metrics

Click **Machine Learning → View Metrics** to review:

| Metric | What it measures |
|---|---|
| Inference Latency | Average ms per image for tag prediction |
| Training Time | Wall-clock seconds per epoch and total |
| Accuracy | Share of correctly predicted tags vs ground truth |
| Precision / Recall / F1 | Per-class and macro-averaged scores |

---

## Configuration

Edit `config.json` (auto-created on first run) or use the in-app **Model Configuration** dialog.

### Key Settings

```jsonc
{
  "hyperparameters": {
    "learning_rate": 0.001,       // SGD/Adam step size
    "batch_size": 32,             // Images per training step
    "epochs": 10,                 // Max training epochs
    "dropout_rate": 0.3,          // Regularization
    "image_size": [224, 224],     // Resize target before inference
    "confidence_threshold": 0.5,  // Minimum score to apply a tag
    "max_suggestions": 5          // Max predicted tags shown per image
  },
  "app_config": {
    "model_type": "resnet18",         // resnet18 | resnet34 | resnet50 | efficientnet_b0 | efficientnet_b1 | mobilenet_v2 | vgg16
    "use_gpu": true,
    "supervised_mode": true,
    "bootstrap_from_folders": true,
    "gallery_columns": 5,
    "thumbnail_size": [120, 120]
  }
}
```

---

## ML Models Supported

All backbones are loaded with ImageNet pre-trained weights and fine-tuned on the user's image collection via a custom classification head (`hidden_layers` → `num_classes`). Set `model_type` in `config.json` to the value in the **Config key** column.

| Backbone | Config key | Params (M) | ImageNet Top-1 (%) | Input Size | Feature Dim | Architecture Family |
|---|---|---|---|---|---|---|
| **ResNet-18** | `resnet18` | 11.7 | 69.8 | 224×224 | 512 | Residual Networks |
| **ResNet-34** | `resnet34` | 21.8 | 73.3 | 224×224 | 512 | Residual Networks |
| **ResNet-50** | `resnet50` | 25.6 | 76.1 | 224×224 | 2048 | Residual Networks |
| **EfficientNet-B0** | `efficientnet_b0` | 5.3 | 77.1 | 224×224 | 1280 | Compound Scaling |
| **EfficientNet-B1** | `efficientnet_b1` | 7.8 | 78.8 | 240×240 | 1280 | Compound Scaling |
| **MobileNet V2** | `mobilenet_v2` | 3.4 | 71.9 | 224×224 | 1280 | Inverted Residuals |
| **VGG-16** | `vgg16` | 138.4 | 71.6 | 224×224 | 4096 | Sequential Convolutions |

> **Recommendation:** Start with `resnet18` (fast, low memory) or `efficientnet_b0` (best accuracy/size trade-off). Use `mobilenet_v2` on memory-constrained machines. Avoid `vgg16` unless you have a GPU — its 138 M parameters make CPU training impractically slow.

---

## Performance Metrics

The table below shows representative results from the included `ExampleImageFolder` dataset (approx. 500 images across 20 categories).

| Model | Epochs | Training Time | Top-1 Accuracy |
|---|---|---|---|
| ResNet-18 | 10 | ~3 min (CPU) | ~82 % |
| EfficientNet-B0 | 10 | ~5 min (CPU) | ~87 % |

Training graphs (loss / accuracy per epoch) are saved automatically to `data/training_graphs/`.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Model won't train | Fewer than 2 tag classes | Add more tagged images or folders |
| Poor predictions | Insufficient training data | Add 20–30 examples per class; retrain |
| Slow inference | CPU only | Enable GPU (`use_gpu: true`) or lower `image_size` |
| Import errors on startup | Missing packages | Re-run `pip install -r requirements.txt` |
| DB locked error | Multiple instances open | Close other application windows |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open a GitHub Issue in this repository.
