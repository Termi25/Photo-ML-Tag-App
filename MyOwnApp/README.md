# ML-Enhanced Image Sorting Application

A sophisticated image management and sorting application built with a 4-tier architecture that leverages machine learning for automatic image categorization and tagging.

## Architecture Overview

The application follows a clean 4-tier architecture:

```
┌─────────────────────────────────────────────────┐
│         TIER 1: User Interface (UI)             │
│  - File Explorer View (Detailed/Gallery)        │
│  - Model Configuration Menu                     │
│  - Tag Management Sidebar                       │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│      TIER 2: Logic Controller                   │
│  - File Watcher                                 │
│  - Tagging Engine                               │
│  - Training Manager                             │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│      TIER 3: ML Module                          │
│  - Image Preprocessor                           │
│  - Inference Engine                             │
│  - Training Loop                                │
│  - Unsupervised Clustering                      │
└─────────────────────────────────────────────────┘
                      ↕
┌─────────────────────────────────────────────────┐
│      TIER 4: Data Layer                         │
│  - SQLite Metadata Store                        │
│  - Model Weights Storage                        │
│  - Image Assets                                 │
└─────────────────────────────────────────────────┘
```

## Features

### Core Functionality
- **File Explorer**: Windows-style interface with detailed and gallery views
- **Image Browsing**: Navigate through image collections with thumbnail previews
- **Tag Management**: Add, remove, and view tags for images
- **Multi-Tag Search**: Boolean search with AND/OR logic

### Machine Learning Capabilities
- **Supervised Learning**: Train models using manual tags or folder structure
- **Unsupervised Learning**: K-means clustering for automatic organization
- **Auto-Tagging**: Predict tags for images using trained models
- **Fine-Tuning**: Continuously improve models with user corrections

### Measurement & Benchmarking
- **Inference Latency**: Track prediction speed per image
- **Training Convergence**: Monitor training time and accuracy
- **Sorting Accuracy**: Compare predictions against ground truth
- **Metrics Tracking**: Detailed performance analytics

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows/Linux/macOS

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd MyOwnApp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support** (optional, significantly faster):
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Run the application**:
   ```bash
   python file_explorer.py
   ```

## Usage Guide

### Initial Setup

1. **Launch the Application**: Run `python file_explorer.py`
2. **Navigate to Image Folder**: Use the path bar or drives list to navigate to your image collection (defaults to `../ExampleImageFolder`)

### Training Your First Model

#### Option 1: Bootstrap from Folder Structure
Best for pre-organized collections:

1. Navigate to your organized image folder structure (e.g., `ExampleImageFolder/ANIME/`, `ExampleImageFolder/VIDEO GAMES/`)
2. Click **Machine Learning → Train from Folders**
3. Wait for training to complete (progress shown in status bar)
4. Model will be saved automatically as "latest"

#### Option 2: Manual Tagging
Best for custom organization:

1. Click on an image in the file view
2. In the **Tag Management** sidebar, enter a tag name
3. Click **Add Manual Tag**
4. Repeat for multiple images
5. Click **Machine Learning → Train from Tags**

### Auto-Sorting Images

Once you have a trained model:

1. Navigate to a directory with unsorted images
2. Click **Machine Learning → Auto-Sort Images**
3. The system will predict and apply tags to all images
4. Review predictions in the Tag Management sidebar
5. Correct any mistakes by adding/removing tags

### Using the Tag Management Sidebar

**Selecting Images**:
- Single-click an image in the file view to select it
- The sidebar updates to show current tags and predictions

**Current Tags**:
- View all tags applied to the selected image
- Shows confidence score and source (manual/model/clustering)
- Click **Remove Tag** to delete a tag

**Adding Tags**:
- Type tag name in the input field
- Click **Add Manual Tag**
- Tags are marked as ground truth for validation

**Predicted Tags**:
- Auto-populated when image is selected (if model is loaded)
- Shows AI predictions with confidence scores
- Click **Apply Selected Prediction** to accept a suggestion

### Model Configuration

Access via **Machine Learning → Model Configuration**:

**Hyperparameters Tab**:
- `learning_rate`: Controls training step size (default: 0.001)
- `batch_size`: Number of images per training batch (default: 32)
- `epochs`: Maximum training iterations (default: 10)
- `dropout_rate`: Regularization (default: 0.3)
- `confidence_threshold`: Minimum confidence for predictions (default: 0.5)

**Application Tab**:
- `use_gpu`: Enable GPU acceleration
- `supervised_mode`: Use supervised vs unsupervised learning
- `bootstrap_from_folders`: Initialize from folder structure

### Unsupervised Clustering

For collections without labels:

1. Click **Machine Learning → Unsupervised Clustering**
2. Enter number of clusters (e.g., 10)
3. System will group similar images automatically
4. Tags will be created as "cluster_0", "cluster_1", etc.
5. Review and rename clusters as needed

### Viewing Metrics

Track system performance:

1. Click **Machine Learning → View Metrics**
2. Review:
   - **Inference Latency**: Time to predict tags
   - **Training Time**: How long training took
   - **Accuracy**: Prediction correctness vs ground truth
   - **Precision/Recall/F1**: Detailed accuracy metrics

## Experimental Workflow

### Phase 1: Initialization
```python
# System automatically:
# 1. Scans target directory
# 2. Registers images in database
# 3. Loads or initializes model
```

### Phase 2: Inference
```python
# When Auto-Sort is clicked:
# 1. Preprocesses images (resize, normalize)
# 2. Runs inference engine
# 3. Generates predictions
# 4. Updates metadata store
```

### Phase 3: Validation & Feedback
```python
# User reviews predictions:
# 1. Compares with ground truth (manual tags)
# 2. Makes corrections via UI
# 3. System logs corrections
```

### Phase 4: Re-training
```python
# Training Manager:
# 1. Collects corrected data
# 2. Initiates training loop
# 3. Performs backpropagation
# 4. Saves updated model weights
# 5. Cycle repeats
```

## Measured Variables

### Independent Variables
- **Hyperparameters**: learning_rate, batch_size, epochs, dropout_rate
- **Model Initialization**: Pre-trained weights vs bootstrapped from folders
- **Training Mode**: Supervised vs unsupervised

### Dependent Variables
1. **Inference Latency**: Time per image batch (milliseconds)
2. **Training Convergence Speed**: Time and epochs to reach target accuracy
3. **Sorting Accuracy**: Percentage of correct predictions

## File Structure

```
MyOwnApp/
├── file_explorer.py         # Main UI application
├── config.py                # Configuration management
├── data_layer.py            # Database and storage
├── logic_controller.py      # Orchestration layer
├── ml_module.py             # ML algorithms
├── metrics.py               # Performance tracking
├── requirements.txt         # Dependencies
├── README.md               # This file
├── config.json             # User configuration (auto-created)
└── data/                   # Auto-created data directory
    ├── image_metadata.db   # SQLite database
    ├── models/             # Trained model weights
    ├── cache/              # Temporary files
    └── metrics.json        # Performance logs
```

## Database Schema

### Images Table
Stores image metadata:
- file_path, file_name, file_size
- created_at, modified_at, added_to_db, last_processed

### Tags Table
Manages tag definitions:
- tag_name, category, color

### Image_Tags Table
Many-to-many relationship:
- image_id, tag_id, confidence, source

### Ground_Truth Table
Validation annotations:
- image_id, tag_id, verified_by, verified_at

## Tips & Best Practices

### For Best Accuracy
- Provide at least 20-30 examples per category
- Use consistent naming for tags
- Regularly review and correct predictions
- Use ground truth annotations for validation

### For Best Performance
- Enable GPU if available
- Use appropriate batch sizes (16-32 for most systems)
- Start with lower resolution images for faster training
- Use pre-trained models when possible

### For Organization
- Use hierarchical tag structure (e.g., "anime/mecha", "anime/slice_of_life")
- Create tag categories for better filtering
- Export/backup your database regularly

## Troubleshooting

### Model won't train
- Ensure you have at least 2 different tags/folders
- Check that images are valid and not corrupted
- Verify sufficient disk space for model weights

### Predictions are poor
- Train longer (increase epochs)
- Add more training examples
- Adjust confidence threshold
- Try different learning rates

### Slow performance
- Enable GPU in configuration
- Reduce batch size
- Lower image resolution in config
- Close other resource-intensive applications

## Advanced Usage

### Custom Models
Modify `ml_module.py` to use different architectures:
- Change `model_type` in config to 'resnet' or 'efficientnet'
- Implement custom CNN architectures

### Automated Scripts
Use the logic_controller module directly:
```python
from logic_controller import logic_controller
logic_controller.initialize('/path/to/images')
logic_controller.bootstrap_from_folders()
```

### Batch Processing
Process multiple directories:
```python
for folder in folders:
    logic_controller.initialize(folder)
    logic_controller.start_initial_sorting()
```

## Credits

Developed as part of doctoral research in machine learning applications for image management and organization.

## License

Academic/Research Use

## Contact

For questions, issues, or contributions, please create an issue in the repository.
