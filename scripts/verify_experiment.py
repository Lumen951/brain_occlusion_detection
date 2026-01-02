"""Verify experiment configuration before training."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import torch
from src.models.pretrained_loader import create_vit_b16_pretrained


def verify_config(config_path: str):
    """Verify experiment configuration."""
    print("=" * 80)
    print("Experiment Configuration Verification")
    print("=" * 80)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n📋 Experiment: {config['experiment']['name']}")
    print(f"📝 Description: {config['experiment']['description']}")

    # Model config
    model_config = config['model']
    print(f"\n🏗️  Model Configuration:")
    print(f"  - Type: {model_config['type']}")
    print(f"  - Pretrained: {model_config['pretrained']}")
    print(f"  - Freeze Backbone: {model_config['freeze_backbone']}")
    print(f"  - Drop Rate: {model_config.get('drop_rate', 0.0)}")
    print(f"  - Drop Path Rate: {model_config.get('drop_path_rate', 0.0)}")

    # Training config
    train_config = config['training']
    print(f"\n🎯 Training Configuration:")
    print(f"  - Epochs: {train_config['epochs']}")
    print(f"  - Learning Rate: {train_config['optimizer']['lr']}")
    print(f"  - Weight Decay: {train_config['optimizer']['weight_decay']}")
    print(f"  - Early Stopping Patience: {train_config['early_stopping']['patience']}")

    # Dataset config
    dataset_config = config['dataset']
    print(f"\n📊 Dataset Configuration:")
    print(f"  - Type: {dataset_config['type']}")
    print(f"  - Batch Size: {dataset_config['batch_size']}")
    print(f"  - Image Size: {dataset_config['image_size']}")

    # Test model creation
    print(f"\n🔬 Creating Model to Verify...")
    model = create_vit_b16_pretrained(
        num_classes=model_config['num_classes'],
        pretrained=False,  # Don't download weights for testing
        freeze_backbone=model_config['freeze_backbone'],
        drop_rate=model_config.get('drop_rate', 0.0),
        drop_path_rate=model_config.get('drop_path_rate', 0.0),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n✅ Model Created Successfully!")
    print(f"  - Total Parameters: {total_params / 1e6:.2f}M")
    print(f"  - Trainable Parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.1f}%)")
    print(f"  - Frozen Parameters: {frozen_params / 1e6:.2f}M ({frozen_params / total_params * 100:.1f}%)")

    # Verify freeze is correct
    if model_config['freeze_backbone']:
        if trainable_params > 2e6:  # Should be ~1.5K for classification head
            print(f"\n⚠️  WARNING: Expected ~0.002M trainable params when frozen, got {trainable_params / 1e6:.2f}M")
        else:
            print(f"\n✅ Freeze configuration verified! Only classification head is trainable.")

    # Calculate data-to-parameter ratio
    train_images = 210  # Estimated from split
    ratio = train_images / trainable_params
    print(f"\n📈 Data-to-Parameter Ratio:")
    print(f"  - Training Images: {train_images}")
    print(f"  - Images per 1K Params: {ratio * 1000:.2f}")
    print(f"  - ImageNet ratio: ~15 images/1K params")

    if ratio * 1000 < 1:
        print(f"  ⚠️  Very low ratio! High overfitting risk.")
    elif ratio * 1000 < 10:
        print(f"  ⚠️  Low ratio. Regularization strongly recommended.")
    else:
        print(f"  ✅ Reasonable ratio for this dataset size.")

    print("\n" + "=" * 80)
    print("Verification Complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify experiment configuration")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vit_b16_image_split.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    config_path = project_root / args.config if not Path(args.config).is_absolute() else args.config
    verify_config(str(config_path))
