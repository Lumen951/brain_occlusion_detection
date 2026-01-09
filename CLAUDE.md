# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a brain occlusion detection research project that compares human and AI vision strategies on occluded aircraft images. The project uses the OIID (Occluded Image Interpretation Dataset) dataset and trains deep learning models (Vision Transformers, CNNs) to classify occluded aircraft images.

**Core research questions:**
- How do different model architectures (ViT, ResNet, Swin, MAE) handle occlusion?
- How do AI models compare to human performance?
- What are the differences in error patterns between architectures?

## Common Commands

### Training Models

```bash
# Train ViT-B/16 with image-based split
python scripts/training/train_model.py --config configs/vit_b16_image_split.yaml

# Train ResNet-50 with image-based split
python scripts/training/train_model.py --config configs/resnet50_image_split.yaml

# Monitor training with TensorBoard
tensorboard --logdir experiments/
```

### Evaluation and Analysis

```bash
# Evaluate model performance by occlusion level
python scripts/evaluation/evaluate_by_occlusion.py \
  --checkpoint experiments/resnet50/image_split/checkpoints/best_model.pth \
  --config configs/resnet50_image_split.yaml

# Comprehensive model comparison
python scripts/analysis/compare_models_comprehensive.py \
  --output-dir experiments/analysis/comprehensive

# Analyze error patterns between models
python scripts/analysis/analyze_errors.py

# Compare human vs AI performance
python scripts/analysis/compare_human_vs_ai_final.py

# Analyze human behavioral data
python scripts/analysis/analyze_human_behavioral_data.py
```

### Data Preparation

**Note:** Image-based splits should already exist in `data/` directory (train, val, test).

If you need to recreate them from the OIID dataset:
```bash
# Check for data preparation scripts in docs/ directory
# The data splits are created from E:/Dataset/ds005226
```

## Architecture Overview

### Data Loading

Two dataset modes are supported:

1. **Stimulus-based dataset** (`src/data/stimulus_dataset.py`):
   - Loads from OIID dataset structure with TSV event files
   - Organized by subjects (train: 50 subjects, val: 8, test: 7)
   - Extracts labels and occlusion levels from TSV metadata

2. **Image-based dataset** (`src/data/image_dataset.py`):
   - Loads from pre-split directories (`data/train`, `data/val`, `data/test`)
   - Parses labels and occlusion levels from filenames (e.g., `Aircraft1_70%_2.jpg`)
   - Used for fair model comparison

**Important:** Image-based split is the current standard for model comparison because it prevents data leakage and ensures true generalization evaluation.

### Model Architecture

Models are created using the `timm` library in `src/models/pretrained_loader.py`:

- **ViT-B/16**: `create_vit_b16_pretrained()` - Vision Transformer base model
- **ResNet-50**: `create_resnet50_pretrained()` - Residual Network

Both support:
- `freeze_backbone`: Freeze backbone layers, only train classification head
- `drop_rate`: Dropout for regularization
- Pretrained ImageNet weights

### Training Pipeline

The `Trainer` class in `scripts/train_model.py` handles:
- Configuration loading from YAML files
- Model creation and optimizer setup
- Training loop with mixed precision (AMP)
- Early stopping based on validation loss
- Checkpoint management
- TensorBoard logging

**Key configuration files:**
- `configs/vit_b16_image_split.yaml`: ViT-B/16 configuration
- `configs/resnet50_image_split.yaml`: ResNet-50 configuration

### Evaluation and Analysis Scripts

Located in `scripts/` directory:

- **`evaluate_by_occlusion.py`**: Evaluates model performance broken down by occlusion level (10%, 70%, 90%)
- **`analyze_errors.py`**: Compares error patterns between models, computes Jaccard similarity
- **`compare_performance.py`**: Generates comparison plots and metrics
- **`extract_attention.py`**: Extracts and visualizes ViT attention maps
- **`load_human_data.py`**: Loads human behavioral data from OIID dataset

## Current Experiment Strategy

The project follows a **progressive model comparison strategy**:

**Phase 1 (Current)**: Core baseline
- ViT-B/16 vs ResNet-50 on image-based split
- Focus on understanding architectural differences
- Error pattern analysis and attention visualization

**Phase 2 (Planned)**: Architecture family expansion
- Add Swin-B (hierarchical Transformer)
- Add MAE-ViT (self-supervised pretraining)
- Add ConvNeXt-B (modern CNN)

**Phase 3 (Optional)**: Full comparison
- 10+ models including hybrid architectures
- Comprehensive analysis for top-tier conference submission

## Configuration Format

All training configurations follow this YAML structure:

```yaml
experiment:
  name: "experiment_name"
  phase: "phase1_baseline"

dataset:
  type: "image_split"  # or "stimulus"
  train_dir: "data/train"
  val_dir: "data/val"
  test_dir: "data/test"
  image_size: 224
  batch_size: 16

model:
  type: "resnet50"  # or "vit_b16"
  num_classes: 2
  pretrained: true
  freeze_backbone: true  # For small datasets
  drop_rate: 0.5

training:
  epochs: 50
  optimizer:
    type: "adamw"
    lr: 3.0e-5
    weight_decay: 0.2
  scheduler:
    type: "cosine"
    warmup_epochs: 5
  early_stopping:
    enabled: true
    patience: 10
```

## Key Implementation Details

### Path Resolution

Scripts automatically resolve relative paths from project root:
```python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

Always place this at the top of scripts before importing `src` modules.

### Data Organization

- **`data/`**: Image-based splits (train, val, test)
- **`experiments/`**: Training outputs organized by model type
  - `experiments/{model_name}/{split_name}/checkpoints/`: Model checkpoints
  - `experiments/{model_name}/{split_name}/logs/`: TensorBoard logs
  - `experiments/{model_name}/{split_name}/metrics/`: Evaluation results

### Regularization Strategy

For small datasets like OIID, the project uses:
- **Freeze backbone**: Only train the final classification layer (reduces trainable params from 86M to ~2K for ResNet-50)
- **Enhanced data augmentation**: Random rotation, affine transforms, color jitter
- **Dropout**: `drop_rate=0.5` on classifier
- **Weight decay**: `0.2` for stronger L2 regularization
- **Lower learning rate**: `3e-5` with cosine annealing

### Windows-Specific Notes

- YAML files must use UTF-8 encoding (specified when loading)
- Use forward slashes in YAML paths or double backslashes
- Python scripts handle path conversion automatically

## Research Workflow

1. **Prepare data**: Use `create_image_splits.py` to generate train/val/test splits
2. **Train models**: Run `train_model.py` with appropriate config
3. **Evaluate**: Use `evaluate_by_occlusion.py` to get per-occlusion metrics
4. **Compare**: Run analysis scripts (error analysis, performance comparison)
5. **Visualize**: Generate plots and attention maps
6. **Decision**: Based on results, decide whether to expand to Phase 2

## Analysis Tools Reference

See `TOOLS_USAGE.md` for detailed usage of all analysis scripts.

**Quick checklist after ResNet-50 training completes:**
```bash
# 1. Evaluate performance
python scripts/evaluate_by_occlusion.py --checkpoint [path] --config [path]

# 2. Analyze errors
python scripts/analyze_errors.py

# 3. Compare performance
python scripts/compare_performance.py

# 4. Extract attention
python scripts/extract_attention.py --compare-errors
```

## Important Constraints

- **Never use relative imports** in scripts without setting `sys.path` first
- **Always check data paths** in configs before training
- **Use freeze_backbone=True** for small datasets to prevent overfitting
- **Validate checkpoints** after training before running analysis
- **Keep experiment naming consistent** (e.g., `resnet50_image_split`)

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Fix**: Ensure `sys.path.insert(0, str(project_root))` is at the top of your script

**Issue**: Training accuracy high but validation accuracy low
- **Fix**: Enable `freeze_backbone=True`, increase `weight_decay`, reduce `lr`

**Issue**: CUDA out of memory
- **Fix**: Reduce `batch_size`, or use smaller model (ViT-Tiny instead of ViT-Base)

**Issue**: Data path not found
- **Fix**: Update paths in config YAML, ensure `data/` directory exists

## Future Development

Based on findings from Phase 1, potential directions:
- Attention mechanism analysis (entropy, spatial distribution)
- Representational Similarity Analysis (RSA) between models
- fMRI data integration (if available)
- Architecture innovations based on discovered patterns
