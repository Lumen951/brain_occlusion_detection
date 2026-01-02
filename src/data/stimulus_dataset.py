"""Dataset for occluded aircraft classification task."""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OccludedAircraftDataset(Dataset):
    """Dataset for occluded aircraft image classification.

    This dataset handles the BIDS-formatted fMRI stimulus dataset with:
    - 2 aircraft classes (Aircraft1=0, Aircraft2=1)
    - 3 occlusion levels (10%, 70%, 90%)
    - TSV event files with image labels
    """

    def __init__(
        self,
        dataset_root: str,
        subject_ids: List[str],
        transform: Optional[Callable] = None,
        occlusion_levels: Optional[List[float]] = None,
    ):
        """
        Args:
            dataset_root: Root directory of the BIDS dataset (e.g., "E:/Dataset/ds005226")
            subject_ids: List of subject IDs to include (e.g., ["01", "02", ...])
            transform: Optional image transforms
            occlusion_levels: Optional filter for specific occlusion levels (e.g., [0.1, 0.7, 0.9])
        """
        self.dataset_root = Path(dataset_root)
        self.stimuli_dir = self.dataset_root / "stimuli"
        self.transform = transform
        self.occlusion_levels = occlusion_levels

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        if not self.stimuli_dir.exists():
            raise FileNotFoundError(f"Stimuli directory not found: {self.stimuli_dir}")

        # Load all event files and extract stimulus information
        self.samples = self._load_samples(subject_ids)

    def _load_samples(self, subject_ids: List[str]) -> List[Dict]:
        """Load samples from TSV event files."""
        samples = []

        for sub_id in subject_ids:
            # Ensure sub_id is formatted as 2-digit string (e.g., 01, 02, ..., 65)
            sub_id_str = f"{sub_id:02d}" if isinstance(sub_id, int) else sub_id
            sub_dir = self.dataset_root / f"sub-{sub_id_str}" / "ses-01" / "func"
            if not sub_dir.exists():
                print(f"Warning: Subject directory not found: {sub_dir}")
                continue

            # Find all event TSV files for task-image runs
            event_files = sorted(sub_dir.glob("*_task-image_run-*_events.tsv"))

            for event_file in event_files:
                df = pd.read_csv(event_file, sep='\t')

                # Filter out rest trials (stim_file == "rest.jpg")
                df = df[df['stim_file'] != 'rest.jpg'].copy()

                for _, row in df.iterrows():
                    stim_file = row['stim_file']
                    label = int(row['stim_lable'])  # Note: dataset has typo "lable"

                    # Extract occlusion level from filename (e.g., "Aircraft1_70%_2.jpg" -> 0.7)
                    # TSV has 0.75 but filename has 70%, so we use filename as ground truth
                    import re
                    match = re.search(r'_(\d+)%_', stim_file)
                    if match:
                        occlusion = int(match.group(1)) / 100.0
                    else:
                        # Fallback to TSV value if filename parsing fails
                        occlusion = float(row['levelOfOcclusion'])

                    # Filter by occlusion level if specified
                    if self.occlusion_levels is not None:
                        if occlusion not in self.occlusion_levels:
                            continue

                    stim_path = self.stimuli_dir / stim_file
                    if stim_path.exists():
                        samples.append({
                            'image_path': str(stim_path),
                            'label': label,
                            'occlusion_level': occlusion,
                            'subject_id': sub_id_str,
                            'stimulus_file': stim_file,
                        })
                    else:
                        print(f"Warning: Stimulus file not found: {stim_path}")

        if len(samples) == 0:
            raise ValueError("No valid samples found in the dataset!")

        return samples

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
            'subject_id': sample['subject_id'],
            'stimulus_file': sample['stimulus_file'],
            'image_path': sample['image_path'],  # For visualization tools
        }

        return image, label, metadata


def get_default_transforms(img_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """Get default image transforms for ViT.

    Args:
        img_size: Target image size (ViT typically uses 224)
        is_training: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_dataloaders(
    dataset_root: str,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: Optional[List[str]] = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    occlusion_levels: Optional[List[float]] = None,
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        dataset_root: Root directory of the BIDS dataset
        train_subjects: List of subject IDs for training (e.g., ["01", "02", ...])
        val_subjects: List of subject IDs for validation
        test_subjects: Optional list of subject IDs for testing
        batch_size: Batch size
        img_size: Image size for ViT (typically 224)
        num_workers: Number of data loading workers
        occlusion_levels: Optional filter for specific occlusion levels

    Returns:
        Dictionary with 'train', 'val', and optionally 'test' dataloaders
    """
    train_transform = get_default_transforms(img_size=img_size, is_training=True)
    val_transform = get_default_transforms(img_size=img_size, is_training=False)

    # Create datasets
    train_dataset = OccludedAircraftDataset(
        dataset_root=dataset_root,
        subject_ids=train_subjects,
        transform=train_transform,
        occlusion_levels=occlusion_levels,
    )

    val_dataset = OccludedAircraftDataset(
        dataset_root=dataset_root,
        subject_ids=val_subjects,
        transform=val_transform,
        occlusion_levels=occlusion_levels,
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

    if test_subjects:
        test_dataset = OccludedAircraftDataset(
            dataset_root=dataset_root,
            subject_ids=test_subjects,
            transform=val_transform,
            occlusion_levels=occlusion_levels,
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
