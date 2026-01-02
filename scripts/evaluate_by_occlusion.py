"""Evaluate model performance by occlusion level."""
import sys
from pathlib import Path

# Setup project root and paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch.utils.data import DataLoader
from src.data.stimulus_dataset import OccludedAircraftDataset, get_default_transforms
from src.models.pretrained_loader import create_vit_b16_pretrained


def evaluate_by_occlusion(checkpoint_path: str, config_path: str):
    """Evaluate model and break down results by occlusion level."""

    # Convert to absolute paths relative to project root
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create test dataset directly
    test_dataset = OccludedAircraftDataset(
        dataset_root=config['dataset']['root'],
        subject_ids=config['dataset']['test_subjects'],
        transform=get_default_transforms(img_size=224, is_training=False),
        occlusion_levels=None  # Use all occlusion levels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vit_b16_pretrained(num_classes=2, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Collect predictions
    all_preds = []
    all_labels = []
    all_occlusions = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels, meta in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_occlusions.extend(meta['occlusion_level'].numpy())

    # Create DataFrame
    df = pd.DataFrame({
        'label': all_labels,
        'pred': all_preds,
        'occlusion': all_occlusions
    })

    # Overall results
    print("\n" + "="*60)
    print("Overall Test Results")
    print("="*60)
    overall_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    print(f"Accuracy:  {overall_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Total samples: {len(all_labels)}")

    # By occlusion level
    print("\n" + "="*60)
    print("Performance by Occlusion Level")
    print("="*60)

    results = []
    for occ_level in sorted(df['occlusion'].unique()):
        subset = df[df['occlusion'] == occ_level]
        acc = accuracy_score(subset['label'], subset['pred'])
        prec, rec, f1, _ = precision_recall_fscore_support(
            subset['label'], subset['pred'], average='binary'
        )

        print(f"\n{round(occ_level*100):2d}% Occlusion:")
        print(f"  Accuracy:  {acc:.4f} ({len(subset)} samples)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        results.append({
            'occlusion_level': f"{round(occ_level*100)}%",
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'num_samples': len(subset)
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_dir = Path(checkpoint_path).parent.parent / 'metrics'
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(output_dir / 'performance_by_occlusion.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'performance_by_occlusion.csv'}")

    # Robustness analysis
    print("\n" + "="*60)
    print("Robustness Analysis")
    print("="*60)

    acc_10 = results_df[results_df['occlusion_level'] == '10%']['accuracy'].values[0]
    acc_90 = results_df[results_df['occlusion_level'] == '90%']['accuracy'].values[0]
    robustness_gap = acc_10 - acc_90

    print(f"10% occlusion accuracy: {acc_10:.4f}")
    print(f"90% occlusion accuracy: {acc_90:.4f}")
    print(f"Robustness gap:         {robustness_gap:.4f}")
    print(f"Relative drop:          {robustness_gap/acc_10*100:.1f}%")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='scripts/experiments/vit_b16/quick_test/checkpoints/best_model.pth',
        help='Path to model checkpoint (relative to project root)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vit_b16_quick_test.yaml',
        help='Path to config file (relative to project root)'
    )
    args = parser.parse_args()

    print(f"Project root: {project_root}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print()

    evaluate_by_occlusion(args.checkpoint, args.config)
