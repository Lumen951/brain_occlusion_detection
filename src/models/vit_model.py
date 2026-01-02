"""ViT model for occluded aircraft classification."""

import torch
import torch.nn as nn
from vit_pytorch import ViT


class ViTClassifier(nn.Module):
    """Vision Transformer for binary aircraft classification.

    This model uses the vit-pytorch library to classify occluded aircraft images.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 2,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        pretrained: bool = False,
    ):
        """
        Args:
            image_size: Input image size (default: 224)
            patch_size: Patch size for ViT (default: 16, gives 14x14 patches)
            num_classes: Number of output classes (2 for binary classification)
            dim: Dimension of transformer embeddings (default: 768 for ViT-Base)
            depth: Number of transformer layers (default: 12 for ViT-Base)
            heads: Number of attention heads (default: 12 for ViT-Base)
            mlp_dim: Dimension of MLP in transformer (default: 3072 for ViT-Base)
            dropout: Dropout rate in transformer
            emb_dropout: Dropout rate for embeddings
            pretrained: Whether to use pretrained weights (requires timm library)
        """
        super().__init__()

        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            channels=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [batch_size, 3, H, W]

        Returns:
            Logits [batch_size, num_classes]
        """
        return self.vit(x)


def create_vit_tiny(num_classes: int = 2, image_size: int = 224) -> ViTClassifier:
    """Create ViT-Tiny model (small, fast training).

    Good for initial experiments and quick iteration.
    """
    return ViTClassifier(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0.1,
        emb_dropout=0.1,
    )


def create_vit_small(num_classes: int = 2, image_size: int = 224) -> ViTClassifier:
    """Create ViT-Small model (balanced size and performance)."""
    return ViTClassifier(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0.1,
        emb_dropout=0.1,
    )


def create_vit_base(num_classes: int = 2, image_size: int = 224) -> ViTClassifier:
    """Create ViT-Base model (standard ViT architecture).

    This is the default ViT architecture from "An Image is Worth 16x16 Words".
    """
    return ViTClassifier(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
    )


def create_vit_large(num_classes: int = 2, image_size: int = 224) -> ViTClassifier:
    """Create ViT-Large model (highest accuracy, slower training)."""
    return ViTClassifier(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0.1,
        emb_dropout=0.1,
    )
