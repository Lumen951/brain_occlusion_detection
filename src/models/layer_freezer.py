"""Utilities for freezing and unfreezing specific layers in neural networks."""
import torch.nn as nn
from typing import List


def freeze_layers(model: nn.Module, layer_indices: List[int], model_type: str = 'vit'):
    """
    Freeze specific transformer blocks or CNN layers.

    Args:
        model: PyTorch model
        layer_indices: List of layer/block indices to freeze
        model_type: 'vit' for Vision Transformer or 'resnet' for ResNet
    """
    if model_type == 'vit':
        # Freeze ViT blocks
        for name, param in model.named_parameters():
            for block_idx in layer_indices:
                if f'blocks.{block_idx}.' in name:
                    param.requires_grad = False
                    break
    elif model_type == 'resnet':
        # Freeze ResNet layers
        for name, param in model.named_parameters():
            for layer_idx in layer_indices:
                if f'layer{layer_idx}.' in name:
                    param.requires_grad = False
                    break
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(f"Frozen {model_type} layers: {layer_indices}")


def unfreeze_layers(model: nn.Module, layer_indices: List[int], model_type: str = 'vit'):
    """
    Unfreeze specific transformer blocks or CNN layers.

    Args:
        model: PyTorch model
        layer_indices: List of layer/block indices to unfreeze
        model_type: 'vit' for Vision Transformer or 'resnet' for ResNet
    """
    if model_type == 'vit':
        # Unfreeze ViT blocks
        for name, param in model.named_parameters():
            for block_idx in layer_indices:
                if f'blocks.{block_idx}.' in name:
                    param.requires_grad = True
                    break
    elif model_type == 'resnet':
        # Unfreeze ResNet layers
        for name, param in model.named_parameters():
            for layer_idx in layer_indices:
                if f'layer{layer_idx}.' in name:
                    param.requires_grad = True
                    break
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(f"Unfrozen {model_type} layers: {layer_indices}")


def freeze_all_except(model: nn.Module, except_names: List[str]):
    """
    Freeze all parameters except those matching the given names.

    Args:
        model: PyTorch model
        except_names: List of parameter name patterns to keep trainable (e.g., ['head', 'classifier'])
    """
    for name, param in model.named_parameters():
        # Check if any exception pattern matches
        should_keep_trainable = any(pattern in name for pattern in except_names)
        param.requires_grad = should_keep_trainable

    print(f"Frozen all parameters except those containing: {except_names}")


def print_trainable_params(model: nn.Module, detailed: bool = False):
    """
    Print trainable parameter statistics.

    Args:
        model: PyTorch model
        detailed: If True, print each trainable parameter
    """
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
        else:
            frozen_params.append((name, param.numel()))

    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)
    total = total_trainable + total_frozen

    print("=" * 70)
    print("TRAINABLE PARAMETER SUMMARY")
    print("=" * 70)
    print(f"Trainable:  {total_trainable:>12,} ({total_trainable/1e6:>6.2f}M) - {total_trainable/total*100:>5.1f}%")
    print(f"Frozen:     {total_frozen:>12,} ({total_frozen/1e6:>6.2f}M) - {total_frozen/total*100:>5.1f}%")
    print(f"Total:      {total:>12,} ({total/1e6:>6.2f}M)")
    print("=" * 70)

    if detailed:
        if trainable_params:
            print("\nTrainable parameters:")
            for name, count in trainable_params:
                print(f"  {name:<50} {count:>10,} ({count/1e6:>6.3f}M)")

        if frozen_params:
            print("\nFrozen parameters:")
            for name, count in frozen_params[:10]:  # Show first 10
                print(f"  {name:<50} {count:>10,} ({count/1e6:>6.3f}M)")
            if len(frozen_params) > 10:
                print(f"  ... and {len(frozen_params) - 10} more frozen parameters")


if __name__ == "__main__":
    # Test with a simple model
    import timm

    print("Testing layer freezing utilities...")
    print("\n1. Creating MAE ViT-B/16 model...")
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=False, num_classes=2)

    print("\n2. Initial state (all trainable):")
    print_trainable_params(model)

    print("\n3. Freezing blocks 0-8 (first 9 blocks):")
    freeze_layers(model, list(range(9)), model_type='vit')
    print_trainable_params(model)

    print("\n4. Unfreezing blocks 6-8 (last 3 of the frozen blocks):")
    unfreeze_layers(model, [6, 7, 8], model_type='vit')
    print_trainable_params(model)

    print("\n5. Freezing all except head:")
    freeze_all_except(model, ['head'])
    print_trainable_params(model)
