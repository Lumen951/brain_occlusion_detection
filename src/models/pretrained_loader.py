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
):
    """
    Create ResNet-50 model with pretrained ImageNet weights.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers

    Returns:
        ResNet-50 model instance
    """
    model = timm.create_model(
        'resnet50',
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Keep final fc layer trainable
                param.requires_grad = False

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
    vit_model = create_vit_b16_pretrained(pretrained=False)  # Don't download for test
    print(f"ViT-B/16 parameters: {sum(p.numel() for p in vit_model.parameters()) / 1e6:.2f}M")

    print("\nTesting ResNet-50 creation...")
    resnet_model = create_resnet50_pretrained(pretrained=False)
    print(f"ResNet-50 parameters: {sum(p.numel() for p in resnet_model.parameters()) / 1e6:.2f}M")

    print("\n" + "="*60)
    list_available_models()
