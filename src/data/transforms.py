"""MONAI transforms for medical image preprocessing and augmentation."""

from typing import List, Tuple
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
)


def get_preprocessing_transforms(
    keys: List[str],
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0),
    intensity_min: float = -175,
    intensity_max: float = 250,
    normalize_to: Tuple[float, float] = (0.0, 1.0),
) -> Compose:
    """
    Get preprocessing transforms for medical images.

    These transforms are deterministic and should be cached using PersistentDataset.

    Args:
        keys: List of keys to transform (e.g., ["image", "label"])
        spacing: Target voxel spacing in mm
        intensity_min: Minimum intensity for windowing
        intensity_max: Maximum intensity for windowing
        normalize_to: Target intensity range after normalization

    Returns:
        Composed transform pipeline
    """
    return Compose([
        LoadImaged(keys=keys, reader="ITKReader"),
        EnsureChannelFirstd(keys=keys),
        Spacingd(
            keys=keys,
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_min,
            a_max=intensity_max,
            b_min=normalize_to[0],
            b_max=normalize_to[1],
            clip=True,
        ),
    ])


def get_training_transforms(
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    samples_per_image: int = 4,
    flip_prob: float = 0.5,
    rotate_range: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    zoom_range: Tuple[float, float] = (0.9, 1.1),
    intensity_shift: float = 0.1,
    intensity_scale: float = 0.1,
) -> Compose:
    """
    Get training-time augmentation transforms.

    These transforms are stochastic and should NOT be cached.

    Args:
        patch_size: Size of patches to crop
        samples_per_image: Number of patches to sample per image
        flip_prob: Probability of random flip
        rotate_range: Range for random rotation (radians)
        zoom_range: Range for random zoom
        intensity_shift: Range for intensity shift
        intensity_scale: Range for intensity scaling

    Returns:
        Composed transform pipeline
    """
    return Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=samples_per_image,
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=flip_prob,
            spatial_axis=[0, 1, 2],
        ),
        RandRotated(
            keys=["image", "label"],
            range_x=rotate_range[0],
            range_y=rotate_range[1],
            range_z=rotate_range[2],
            prob=0.5,
            mode=("bilinear", "nearest"),
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=zoom_range[0],
            max_zoom=zoom_range[1],
            prob=0.5,
            mode=("trilinear", "nearest"),
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=intensity_shift,
            prob=0.5,
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=intensity_scale,
            prob=0.5,
        ),
        ToTensord(keys=["image", "label"]),
    ])


def get_validation_transforms() -> Compose:
    """
    Get validation-time transforms.

    No augmentation, just conversion to tensors.

    Returns:
        Composed transform pipeline
    """
    return Compose([
        ToTensord(keys=["image", "label"]),
    ])
