"""Extract and visualize attention maps from Vision Transformer."""
import sys
from pathlib import Path

# Setup project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict
from tqdm import tqdm

from src.data.stimulus_dataset import OccludedAircraftDataset, get_default_transforms
from src.models.pretrained_loader import create_vit_b16_pretrained


class AttentionExtractor:
    """Extract attention maps from ViT model."""

    def __init__(self, checkpoint_path: str, config_path: str):
        """
        Initialize attention extractor.

        Args:
            checkpoint_path: Path to ViT checkpoint
            config_path: Path to config file
        """
        # Convert to absolute paths
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path

        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        print("Loading ViT model...")
        self.model = create_vit_b16_pretrained(num_classes=2, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Register hooks to capture attention
        self.attention_maps = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def hook_fn(module, input, output):
            # For ViT, attention is in the output of attention layers
            # timm's ViT returns attention weights when using forward features
            self.attention_maps.append(output)

        # Register hooks on all attention blocks
        for name, module in self.model.named_modules():
            if 'attn' in name and 'drop' not in name:
                module.register_forward_hook(hook_fn)

    def extract_attention(
        self,
        image: torch.Tensor,
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        Extract attention map for a single image.

        Args:
            image: Input image tensor [1, 3, H, W]
            layer_idx: Which transformer layer to extract (-1 for last layer)

        Returns:
            Attention map [num_patches, num_patches]
        """
        self.attention_maps = []

        with torch.no_grad():
            # Get model output and attention
            # For timm ViT, we need to use a different approach
            # to get attention weights

            # Forward pass through blocks and capture attention
            x = self.model.patch_embed(image)

            # Add cls token
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.model.pos_drop(x + self.model.pos_embed)

            # Store attention weights from each block
            attentions = []
            for blk in self.model.blocks:
                # Get attention weights from the attention module
                B, N, C = x.shape
                qkv = blk.attn.qkv(blk.norm1(x)).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)
                attentions.append(attn.cpu().numpy())

                # Continue forward pass
                x = blk(x)

        # Get attention from specified layer
        attention = attentions[layer_idx]  # [batch, num_heads, num_patches, num_patches]

        # Average over heads
        attention = attention.mean(axis=1)[0]  # [num_patches, num_patches]

        return attention

    def visualize_attention(
        self,
        image: Image.Image,
        attention: np.ndarray,
        save_path: str = None,
        head_fusion: str = "mean"
    ) -> np.ndarray:
        """
        Visualize attention map overlaid on image.

        Args:
            image: Original PIL image
            attention: Attention weights [num_patches, num_patches]
            save_path: Path to save visualization
            head_fusion: How to combine multiple heads ("mean", "max", "min")

        Returns:
            Visualization as numpy array
        """
        # Get attention from CLS token to all patches
        cls_attention = attention[0, 1:]  # Skip CLS token itself

        # Reshape to 2D grid
        num_patches = int(np.sqrt(len(cls_attention)))
        attention_map = cls_attention.reshape(num_patches, num_patches)

        # Resize to image size
        img_size = image.size
        attention_resized = Image.fromarray(attention_map).resize(img_size, resample=Image.BILINEAR)
        attention_resized = np.array(attention_resized)

        # Normalize
        attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Attention map
        im = axes[1].imshow(attention_resized, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return attention_resized

    def extract_batch_attention(
        self,
        dataset: OccludedAircraftDataset,
        output_dir: str,
        num_samples: int = None,
        layer_idx: int = -1
    ):
        """
        Extract attention maps for a batch of images.

        Args:
            dataset: Dataset to extract from
            output_dir: Output directory for visualizations
            num_samples: Number of samples to process (None for all)
            layer_idx: Which layer to extract attention from
        """
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        num_samples = num_samples or len(dataset)
        num_samples = min(num_samples, len(dataset))

        print(f"\nExtracting attention maps for {num_samples} samples...")
        print(f"Output directory: {output_dir}")

        for i in tqdm(range(num_samples)):
            image_tensor, label, meta = dataset[i]

            # Get original image
            img_path = Path(meta['image_path'])
            original_image = Image.open(img_path).convert('RGB')

            # Prepare input
            image_input = image_tensor.unsqueeze(0).to(self.device)

            # Extract attention
            attention = self.extract_attention(image_input, layer_idx=layer_idx)

            # Create filename
            stimulus_file = meta['stimulus_file']
            save_name = f"{stimulus_file.replace('.jpg', '')}_attention.png"
            save_path = output_dir / save_name

            # Visualize and save
            self.visualize_attention(original_image, attention, save_path=save_path)

        print(f"\nAttention maps saved to: {output_dir}")

    def compare_correct_vs_error(
        self,
        error_df_path: str,
        dataset: OccludedAircraftDataset,
        output_dir: str,
        num_correct: int = 10,
        num_error: int = 10,
        layer_idx: int = -1
    ):
        """
        Compare attention patterns between correct and incorrect predictions.

        Args:
            error_df_path: Path to error analysis CSV
            dataset: Dataset
            output_dir: Output directory
            num_correct: Number of correct samples to visualize
            num_error: Number of error samples to visualize
            layer_idx: Which layer to extract attention from
        """
        import pandas as pd

        # Load error analysis
        error_df_path = Path(error_df_path)
        if not error_df_path.is_absolute():
            error_df_path = project_root / error_df_path

        df = pd.read_csv(error_df_path)

        # Get correct and error samples
        correct_samples = df[df['vit_error'] == 0].sample(min(num_correct, len(df[df['vit_error'] == 0])))
        error_samples = df[df['vit_error'] == 1].sample(min(num_error, len(df[df['vit_error'] == 1])))

        # Create output directories
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

        correct_dir = output_dir / 'correct'
        error_dir = output_dir / 'error'
        correct_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        print("\nExtracting attention for correct predictions...")
        self._extract_for_samples(correct_samples, dataset, correct_dir, layer_idx)

        print("\nExtracting attention for incorrect predictions...")
        self._extract_for_samples(error_samples, dataset, error_dir, layer_idx)

        print(f"\nComparison saved to: {output_dir}")

    def _extract_for_samples(
        self,
        samples_df,
        dataset: OccludedAircraftDataset,
        output_dir: Path,
        layer_idx: int
    ):
        """Helper to extract attention for specific samples."""
        for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df)):
            stimulus_file = row['stimulus_file']

            # Find sample in dataset
            sample_idx = None
            for i, (_, _, meta) in enumerate(dataset):
                if meta['stimulus_file'] == stimulus_file:
                    sample_idx = i
                    break

            if sample_idx is None:
                continue

            # Extract attention
            image_tensor, label, meta = dataset[sample_idx]
            img_path = Path(meta['image_path'])
            original_image = Image.open(img_path).convert('RGB')

            image_input = image_tensor.unsqueeze(0).to(self.device)
            attention = self.extract_attention(image_input, layer_idx=layer_idx)

            # Save
            save_name = f"{stimulus_file.replace('.jpg', '')}_attention.png"
            save_path = output_dir / save_name
            self.visualize_attention(original_image, attention, save_path=save_path)


def main():
    """Extract attention maps from ViT."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract ViT attention maps")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='scripts/experiments/vit_b16/quick_test/checkpoints/best_model.pth',
        help='Path to ViT checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vit_b16_quick_test.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scripts/experiments/analysis/attention',
        help='Output directory for attention visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=-1,
        help='Which transformer layer to extract (-1 for last)'
    )
    parser.add_argument(
        '--compare-errors',
        action='store_true',
        help='Compare attention for correct vs error samples'
    )
    parser.add_argument(
        '--error-csv',
        type=str,
        default='scripts/experiments/analysis/error_analysis/error_samples.csv',
        help='Path to error analysis CSV'
    )
    args = parser.parse_args()

    print("="*60)
    print("ViT Attention Extraction")
    print("="*60)

    # Initialize extractor
    extractor = AttentionExtractor(args.checkpoint, args.config)

    # Create test dataset
    test_dataset = OccludedAircraftDataset(
        dataset_root=extractor.config['dataset']['root'],
        subject_ids=extractor.config['dataset']['test_subjects'],
        transform=get_default_transforms(img_size=224, is_training=False),
        occlusion_levels=None
    )

    if args.compare_errors:
        # Compare correct vs error samples
        extractor.compare_correct_vs_error(
            error_df_path=args.error_csv,
            dataset=test_dataset,
            output_dir=args.output_dir,
            num_correct=10,
            num_error=10,
            layer_idx=args.layer
        )
    else:
        # Extract for random samples
        extractor.extract_batch_attention(
            dataset=test_dataset,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            layer_idx=args.layer
        )

    print("\n" + "="*60)
    print("Attention extraction completed!")
    print("="*60)


if __name__ == "__main__":
    main()
