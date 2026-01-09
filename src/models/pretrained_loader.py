"""Pretrained model loader using timm library."""
import timm
import torch.nn as nn


def create_vit_b16_pretrained(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    """
    Create ViT-B/16 model with pretrained ImageNet weights.

    Args:
        num_classes: Number of output classes (default: 2 for Aircraft1 vs Aircraft2)
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers (only train head)
        drop_rate: Dropout rate for classifier head
        drop_path_rate: DropPath rate for transformer blocks

    Returns:
        ViT-B/16 model instance
    """
    # Load ViT-B/16 from timm
    # Model: 224x224 input, patch size 16x16, 86M parameters
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    # Optional: Freeze backbone layers (only train classification head)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'head' not in name:  # Keep head trainable
                param.requires_grad = False

        print(f"Frozen backbone. Trainable parameters: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    return model


def create_resnet50_pretrained(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_rate: float = 0.0,
    drop_block_rate: float = 0.0,
):
    """
    Create ResNet-50 model with pretrained ImageNet weights.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        drop_rate: Dropout rate for classifier
        drop_block_rate: DropBlock rate for convolutional blocks

    Returns:
        ResNet-50 model instance
    """
    model = timm.create_model(
        'resnet50',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_block_rate=drop_block_rate,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Keep final fc layer trainable
                param.requires_grad = False

    return model


def create_mae_vit_base_pretrained(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    freeze_layers: list = None,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    """
    Create MAE-pretrained ViT-B/16 model.

    MAE (Masked Autoencoder) is pretrained with 75% random masking on ImageNet,
    making it naturally suited for occlusion-robust vision tasks.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to load MAE pretrained weights
        freeze_backbone: Whether to freeze all backbone layers (only train head)
        freeze_layers: Specific block indices to freeze (e.g., [0,1,2,3,4,5,6,7,8])
                      If provided, overrides freeze_backbone
        drop_rate: Dropout rate for classifier head
        drop_path_rate: DropPath rate for transformer blocks

    Returns:
        MAE ViT-B/16 model instance
    """
    # Load MAE-pretrained ViT-B/16
    # Model: 224x224 input, patch size 16x16, 86M parameters
    # Pretrained with 75% masking on ImageNet-1K
    model = timm.create_model(
        'vit_base_patch16_224.mae',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    # Handle layer freezing
    if freeze_layers is not None:
        # Freeze specific transformer blocks
        for name, param in model.named_parameters():
            # Check if this parameter belongs to a block we want to freeze
            for block_idx in freeze_layers:
                if f'blocks.{block_idx}.' in name:
                    param.requires_grad = False
                    break
            # Keep head always trainable
            if 'head' in name:
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Frozen blocks {freeze_layers}. Trainable: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M")

    elif freeze_backbone:
        # Freeze all backbone layers (only train classification head)
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen backbone. Trainable parameters: {trainable_params / 1e6:.2f}M")

    return model


def list_available_models():
    """List all available pretrained models in timm."""
    print("Available pretrained models:")
    print("\nViT models:")
    vit_models = timm.list_models('vit_*', pretrained=True)
    for model_name in vit_models[:10]:  # Show first 10
        print(f"  - {model_name}")

    print("\nResNet models:")
    resnet_models = timm.list_models('resnet*', pretrained=True)
    for model_name in resnet_models[:10]:
        print(f"  - {model_name}")

    print(f"\nTotal available models: {len(timm.list_models(pretrained=True))}")


if __name__ == "__main__":
    # Test model creation
    print("Testing ViT-B/16 creation...")
    vit_model = create_vit_b16_pretrained(pretrained=False)
    print(f"ViT-B/16 parameters: {sum(p.numel() for p in vit_model.parameters()) / 1e6:.2f}M")

    print("\nTesting ResNet-50 creation...")
    resnet_model = create_resnet50_pretrained(pretrained=False)
    print(f"ResNet-50 parameters: {sum(p.numel() for p in resnet_model.parameters()) / 1e6:.2f}M")

    print("\nTesting MAE ViT-B/16 creation...")
    mae_model = create_mae_vit_base_pretrained(pretrained=False)
    print(f"MAE ViT-B/16 parameters: {sum(p.numel() for p in mae_model.parameters()) / 1e6:.2f}M")

    print("\n" + "="*60)
    list_available_models()
