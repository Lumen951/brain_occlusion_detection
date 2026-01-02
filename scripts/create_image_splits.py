"""Create train/val/test splits by images (not by subjects)."""
import sys
from pathlib import Path

# Setup project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import shutil
import json
from collections import defaultdict
from datetime import datetime


def create_image_splits(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    copy_files: bool = True
):
    """
    Create stratified train/val/test splits by images.

    Args:
        source_dir: Source directory with all images (e.g., E:/Dataset/ds005226/stimuli)
        output_dir: Output directory for splits (e.g., data/)
        train_ratio: Proportion for training set (default: 0.70)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_seed: Random seed for reproducibility
        copy_files: If True, copy files; if False, create symlinks
    """
    np.random.seed(random_seed)

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    splits_dir = output_dir / 'splits'

    for dir_path in [train_dir, val_dir, test_dir, splits_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Creating Image-Based Train/Val/Test Splits")
    print("="*60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {random_seed}")
    print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print()

    # Step 1: Collect all aircraft images (exclude rest.jpg and other files)
    all_images = sorted([f for f in source_dir.glob('Aircraft*.jpg')])

    if len(all_images) == 0:
        raise ValueError(f"No aircraft images found in {source_dir}")

    print(f"Total images found: {len(all_images)}")

    # Step 2: Group images by category (aircraft_type × occlusion_level)
    categories = defaultdict(list)

    for img_path in all_images:
        filename = img_path.name
        # Extract category: Aircraft1_10% → "Aircraft1_10%"
        parts = filename.split('_')
        if len(parts) >= 2:
            aircraft = parts[0]  # Aircraft1 or Aircraft2
            occlusion = parts[1]  # 10%, 70%, or 90%
            category = f"{aircraft}_{occlusion}"
            categories[category].append(img_path)

    print("\nCategory distribution:")
    for cat, images in sorted(categories.items()):
        print(f"  {cat:20s}: {len(images):3d} images")

    # Step 3: Stratified split within each category
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    print("\nPerforming stratified split...")
    for category, images in sorted(categories.items()):
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Remaining goes to test

        # Shuffle images within category
        shuffled_indices = np.random.permutation(n_total)

        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train:n_train + n_val]
        test_indices = shuffled_indices[n_train + n_val:]

        splits['train'].extend([images[i] for i in train_indices])
        splits['val'].extend([images[i] for i in val_indices])
        splits['test'].extend([images[i] for i in test_indices])

        print(f"  {category:20s}: Train={n_train:2d}, Val={n_val:2d}, Test={n_test:2d}")

    # Step 4: Copy/link files to respective directories
    print("\nCopying files to split directories...")

    operation = "Copying" if copy_files else "Linking"

    for split_name, image_list in splits.items():
        target_dir = output_dir / split_name
        print(f"\n{operation} {len(image_list)} images to {split_name}/")

        for img_path in image_list:
            target_path = target_dir / img_path.name

            if copy_files:
                shutil.copy2(img_path, target_path)
            else:
                # Create symbolic link (Windows requires admin or developer mode)
                if target_path.exists():
                    target_path.unlink()
                target_path.symlink_to(img_path)

    # Step 5: Save split metadata
    print("\nSaving split metadata...")

    # Save image lists
    for split_name, image_list in splits.items():
        list_file = splits_dir / f'{split_name}_images.txt'
        with open(list_file, 'w') as f:
            for img_path in sorted(image_list):
                f.write(f"{img_path.name}\n")
        print(f"  Saved {list_file}")

    # Save split info JSON
    split_info = {
        'created_at': datetime.now().isoformat(),
        'random_seed': random_seed,
        'source_directory': str(source_dir),
        'total_images': len(all_images),
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'counts': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        },
        'categories': {
            cat: len(images) for cat, images in sorted(categories.items())
        }
    }

    info_file = splits_dir / 'split_info.json'
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved {info_file}")

    # Step 6: Validation
    print("\n" + "="*60)
    print("Validation")
    print("="*60)

    # Check total count
    total_split = len(splits['train']) + len(splits['val']) + len(splits['test'])
    print(f"Total images in splits: {total_split}")
    print(f"Total images in source: {len(all_images)}")
    assert total_split == len(all_images), "ERROR: Image count mismatch!"
    print("✓ Total count matches")

    # Check no overlap
    train_set = set(img.name for img in splits['train'])
    val_set = set(img.name for img in splits['val'])
    test_set = set(img.name for img in splits['test'])

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    assert len(overlap_train_val) == 0, f"ERROR: Train/Val overlap: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"ERROR: Train/Test overlap: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"ERROR: Val/Test overlap: {overlap_val_test}"
    print("✓ No overlap between splits")

    # Check category balance
    print("\nCategory distribution check:")
    for category in sorted(categories.keys()):
        train_count = sum(1 for img in splits['train'] if category in img.name)
        val_count = sum(1 for img in splits['val'] if category in img.name)
        test_count = sum(1 for img in splits['test'] if category in img.name)

        print(f"  {category:20s}: Train={train_count:2d}, Val={val_count:2d}, Test={test_count:2d}")

    print("\n" + "="*60)
    print("Split creation completed successfully!")
    print("="*60)
    print(f"\nDataset locations:")
    print(f"  Train: {train_dir} ({len(splits['train'])} images)")
    print(f"  Val:   {val_dir} ({len(splits['val'])} images)")
    print(f"  Test:  {test_dir} ({len(splits['test'])} images)")
    print(f"\nMetadata: {splits_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Create image-based train/val/test splits")
    parser.add_argument(
        '--source-dir',
        type=str,
        default='E:/Dataset/ds005226/stimuli',
        help='Source directory with all images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for splits (relative to project root)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Training set ratio (default: 0.70)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symbolic links instead of copying files'
    )
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Convert output_dir to absolute path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    create_image_splits(
        source_dir=args.source_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        copy_files=not args.symlink
    )


if __name__ == "__main__":
    main()
