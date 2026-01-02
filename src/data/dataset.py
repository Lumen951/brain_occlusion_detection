"""Brain occlusion dataset using MONAI."""

from pathlib import Path
from typing import List, Dict, Optional
import yaml

from monai.data import PersistentDataset, DataLoader, list_data_collate
from .transforms import get_preprocessing_transforms, get_training_transforms, get_validation_transforms


class BrainDataset:
    """Dataset manager for brain occlusion detection."""

    def __init__(
        self,
        data_config_path: str = "configs/data_config.yaml",
        train_config_path: str = "configs/train_config.yaml",
    ):
        """
        Initialize dataset manager.

        Args:
            data_config_path: Path to data configuration file
            train_config_path: Path to training configuration file
        """
        with open(data_config_path, 'r') as f:
            self.data_config = yaml.safe_load(f)

        with open(train_config_path, 'r') as f:
            self.train_config = yaml.safe_load(f)

        self.dataset_root = Path(self.data_config['dataset_root'])
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.dataset_root}\n"
                f"Please update 'dataset_root' in {data_config_path}"
            )

    def _get_data_dicts(self, split: str) -> List[Dict[str, str]]:
        """
        Get list of image-label pairs for a split.

        Args:
            split: Data split ('train', 'val', or 'test')

        Returns:
            List of dictionaries with 'image' and 'label' paths
        """
        split_dir = self.dataset_root / self.data_config[f'{split}_split']
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        image_suffix = self.data_config['image_suffix']
        label_suffix = self.data_config['label_suffix']

        data_dicts = []
        for image_path in sorted(split_dir.glob(f"*{image_suffix}")):
            label_path = image_path.parent / image_path.name.replace(
                image_suffix, label_suffix
            )
            if label_path.exists():
                data_dicts.append({
                    "image": str(image_path),
                    "label": str(label_path),
                })

        if len(data_dicts) == 0:
            raise ValueError(
                f"No data found in {split_dir}\n"
                f"Looking for files matching: *{image_suffix} and *{label_suffix}"
            )

        return data_dicts

    def get_train_loader(self) -> DataLoader:
        """
        Get training data loader with caching and augmentation.

        Returns:
            PyTorch DataLoader for training
        """
        data_dicts = self._get_data_dicts('train')

        # Preprocessing transforms (will be cached)
        pre_transforms = get_preprocessing_transforms(
            keys=["image", "label"],
            spacing=tuple(self.train_config['preprocessing']['spacing']),
            intensity_min=self.train_config['preprocessing']['intensity_min'],
            intensity_max=self.train_config['preprocessing']['intensity_max'],
            normalize_to=tuple(self.train_config['preprocessing']['normalize_to']),
        )

        # Augmentation transforms (not cached)
        aug_config = self.train_config['augmentation']
        if aug_config['use_augmentation']:
            aug_transforms = get_training_transforms(
                patch_size=tuple(self.train_config['patch_size']),
                samples_per_image=self.train_config['samples_per_image'],
                flip_prob=aug_config['random_flip_prob'],
                rotate_range=tuple(aug_config['random_rotate_range']),
                zoom_range=tuple(aug_config['random_zoom_range']),
                intensity_shift=aug_config['random_intensity_shift'],
                intensity_scale=aug_config['random_intensity_scale'],
            )
        else:
            aug_transforms = get_validation_transforms()

        # Create persistent dataset with caching
        cache_dir = self.data_config.get('cache_dir', './cache')
        dataset = PersistentDataset(
            data=data_dicts,
            transform=pre_transforms,
            cache_dir=cache_dir,
        )

        # Apply augmentation after caching
        from monai.data import CacheDataset
        aug_dataset = CacheDataset(
            data=dataset,
            transform=aug_transforms,
            cache_rate=0.0,  # Don't cache augmentation
        )

        return DataLoader(
            aug_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
            collate_fn=list_data_collate,
            pin_memory=self.data_config.get('pin_memory', True),
        )

    def get_val_loader(self) -> DataLoader:
        """
        Get validation data loader.

        Returns:
            PyTorch DataLoader for validation
        """
        data_dicts = self._get_data_dicts('val')

        # Preprocessing only (cached)
        pre_transforms = get_preprocessing_transforms(
            keys=["image", "label"],
            spacing=tuple(self.train_config['preprocessing']['spacing']),
            intensity_min=self.train_config['preprocessing']['intensity_min'],
            intensity_max=self.train_config['preprocessing']['intensity_max'],
            normalize_to=tuple(self.train_config['preprocessing']['normalize_to']),
        )

        cache_dir = self.data_config.get('cache_dir', './cache')
        dataset = PersistentDataset(
            data=data_dicts,
            transform=pre_transforms,
            cache_dir=cache_dir,
        )

        # Add tensor conversion
        val_transforms = get_validation_transforms()
        from monai.data import CacheDataset
        val_dataset = CacheDataset(
            data=dataset,
            transform=val_transforms,
            cache_rate=0.0,
        )

        return DataLoader(
            val_dataset,
            batch_size=1,  # Full volume for validation
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            collate_fn=list_data_collate,
            pin_memory=self.data_config.get('pin_memory', True),
        )
