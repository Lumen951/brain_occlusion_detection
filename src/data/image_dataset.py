"""Image-based dataset loader for train/val/test splits."""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable
import re


class ImageSplitDataset(Dataset):
    """Dataset loader for image-based train/val/test splits.

    Loads images directly from split directories (train/val/test).
    Extracts labels and occlusion levels from filenames.
    """

    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            image_dir: Directory containing images (e.g., "data/train")
            transform: Optional image transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Collect all aircraft images
        self.samples = []
        for img_path in sorted(self.image_dir.glob('Aircraft*.jpg')):
            # Parse filename: Aircraft1_70%_2.jpg
            filename = img_path.name

            # Extract aircraft type (0 or 1)
            if filename.startswith('Aircraft1'):
                label = 0
            elif filename.startswith('Aircraft2'):
                label = 1
            else:
                print(f"Warning: Skipping unknown aircraft type: {filename}")
                continue

            # Extract occlusion level from filename
            match = re.search(r'_(\d+)%_', filename)
            if match:
                occlusion = int(match.group(1)) / 100.0
            else:
                print(f"Warning: Cannot parse occlusion level from: {filename}")
                continue

            self.samples.append({
                'image_path': str(img_path),
                'label': label,
                'occlusion_level': occlusion,
                'filename': filename,
            })

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.image_dir}")

        print(f"Loaded {len(self.samples)} images from {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Returns:
            image: Transformed image tensor
            label: Aircraft class (0 or 1)
            metadata: Dict with additional information
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = sample['label']

        # Metadata for tracking
        metadata = {
            'occlusion_level': sample['occlusion_level'],
            'filename': sample['filename'],
            'image_path': sample['image_path'],
        }

        return image, label, metadata


def create_image_split_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """Create dataloaders for image-based splits.

    Args:
        train_dir: Training images directory
        val_dir: Validation images directory
        test_dir: Optional test images directory
        batch_size: Batch size
        img_size: Image size for transforms
        num_workers: Number of data loading workers
        train_transform: Optional custom training transforms
        val_transform: Optional custom validation transforms

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    # Import here to avoid circular dependency
    from torchvision import transforms
    import sys

    # Convert relative paths to absolute (relative to project root)
    project_root = Path(__file__).parent.parent.parent

    train_dir = Path(train_dir)
    if not train_dir.is_absolute():
        train_dir = project_root / train_dir

    val_dir = Path(val_dir)
    if not val_dir.is_absolute():
        val_dir = project_root / val_dir

    if test_dir:
        test_dir = Path(test_dir)
        if not test_dir.is_absolute():
            test_dir = project_root / test_dir

    # Default transforms if not provided
    if train_transform is None:
        # Enhanced augmentation for small dataset to reduce overfitting
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),                    # ← NEW: Random rotation
            transforms.RandomAffine(                          # ← NEW: Random affine
                degrees=0,
                translate=(0.1, 0.1),                        # Slight translation
                scale=(0.9, 1.1)                             # Slight scale variation
            ),
            transforms.ColorJitter(                           # ← ENHANCED
                brightness=0.3,                              # Was 0.1
                contrast=0.3,                                # Was 0.1
                saturation=0.2                               # New
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Create datasets
    train_dataset = ImageSplitDataset(
        image_dir=str(train_dir),
        transform=train_transform,
    )

    val_dataset = ImageSplitDataset(
        image_dir=str(val_dir),
        transform=val_transform,
    )

    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    # Optional test dataloader
    if test_dir:
        test_dataset = ImageSplitDataset(
            image_dir=str(test_dir),
            transform=val_transform,
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
