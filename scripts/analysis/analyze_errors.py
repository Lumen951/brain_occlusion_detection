"""Analyze and compare error patterns between different models."""
import sys
from pathlib import Path

# Setup project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.data.stimulus_dataset import OccludedAircraftDataset, get_default_transforms
from src.models.pretrained_loader import create_vit_b16_pretrained, create_resnet50_pretrained, create_mae_vit_base_pretrained


class ErrorAnalyzer:
    """Analyze error patterns for model comparison."""

    def __init__(self, config_path: str):
        """
        Initialize error analyzer.

        Args:
            config_path: Path to dataset config file
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_predictions(
        self,
        checkpoint_path: str,
        model_type: str = 'vit_b16'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load model predictions on test set.

        Args:
            checkpoint_path: Path to model checkpoint
            model_type: 'vit_b16' or 'resnet50'

        Returns:
            predictions, labels, occlusion_levels, stimulus_files
        """
        # Convert to absolute path
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path

        # Create test dataset
        test_dataset = OccludedAircraftDataset(
            dataset_root=self.config['dataset']['root'],
            subject_ids=self.config['dataset']['test_subjects'],
            transform=get_default_transforms(img_size=224, is_training=False),
            occlusion_levels=None
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
        )

        # Load model
        if model_type == 'vit_b16':
            model = create_vit_b16_pretrained(num_classes=2, pretrained=False)
        elif model_type == 'resnet50':
            model = create_resnet50_pretrained(num_classes=2, pretrained=False)
        elif model_type == 'mae_vit_base':
            model = create_mae_vit_base_pretrained(num_classes=2, pretrained=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # Collect predictions
        all_preds = []
        all_labels = []
        all_occlusions = []
        all_files = []

        print(f"\nLoading predictions for {model_type}...")
        with torch.no_grad():
            for images, labels, meta in tqdm(test_loader):
                images = images.to(self.device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_occlusions.extend(meta['occlusion_level'].numpy())
                all_files.extend(meta['stimulus_file'])

        return (
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_occlusions),
            all_files
        )

    def compute_error_sets(
        self,
        model_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]]
    ) -> pd.DataFrame:
        """
        Compute error sets for multiple models.

        Args:
            model_results: Dict mapping model_name -> (preds, labels, occlusions, files)

        Returns:
            DataFrame with columns:
            - stimulus_file
            - label
            - occlusion_level
            - {model_name}_pred (for each model)
            - {model_name}_error (for each model)
        """
        # Use first model to get base info
        first_model = list(model_results.keys())[0]
        _, labels, occlusions, files = model_results[first_model]

        # Create base dataframe
        df = pd.DataFrame({
            'stimulus_file': files,
            'label': labels,
            'occlusion_level': occlusions
        })

        # Add predictions and error flags for each model
        for model_name, (preds, _, _, _) in model_results.items():
            df[f'{model_name}_pred'] = preds
            df[f'{model_name}_error'] = (preds != labels).astype(int)

        return df

    def analyze_error_overlap(
        self,
        df: pd.DataFrame,
        model1: str,
        model2: str,
        by_occlusion: bool = True
    ) -> Dict:
        """
        Analyze error overlap between two models.

        Args:
            df: DataFrame from compute_error_sets()
            model1: Name of first model
            model2: Name of second model
            by_occlusion: Whether to compute per-occlusion-level stats

        Returns:
            Dictionary with overlap statistics
        """
        results = {}

        # Overall statistics
        error1 = df[f'{model1}_error'] == 1
        error2 = df[f'{model2}_error'] == 1

        both_correct = (~error1) & (~error2)
        only_model1_error = error1 & (~error2)
        only_model2_error = (~error1) & error2
        both_error = error1 & error2

        total = len(df)

        results['overall'] = {
            'total_samples': total,
            f'{model1}_errors': error1.sum(),
            f'{model2}_errors': error2.sum(),
            'both_correct': both_correct.sum(),
            f'only_{model1}_error': only_model1_error.sum(),
            f'only_{model2}_error': only_model2_error.sum(),
            'both_error': both_error.sum(),
            'jaccard_similarity': both_error.sum() / (error1 | error2).sum() if (error1 | error2).sum() > 0 else 0,
            'error_correlation': np.corrcoef(df[f'{model1}_error'], df[f'{model2}_error'])[0, 1]
        }

        # By occlusion level
        if by_occlusion:
            results['by_occlusion'] = {}
            for occ_level in sorted(df['occlusion_level'].unique()):
                subset = df[df['occlusion_level'] == occ_level]

                error1_sub = subset[f'{model1}_error'] == 1
                error2_sub = subset[f'{model2}_error'] == 1

                both_correct_sub = (~error1_sub) & (~error2_sub)
                only_model1_error_sub = error1_sub & (~error2_sub)
                only_model2_error_sub = (~error1_sub) & error2_sub
                both_error_sub = error1_sub & error2_sub

                results['by_occlusion'][f'{int(occ_level*100)}%'] = {
                    'total_samples': len(subset),
                    f'{model1}_errors': error1_sub.sum(),
                    f'{model2}_errors': error2_sub.sum(),
                    'both_correct': both_correct_sub.sum(),
                    f'only_{model1}_error': only_model1_error_sub.sum(),
                    f'only_{model2}_error': only_model2_error_sub.sum(),
                    'both_error': both_error_sub.sum(),
                    'jaccard_similarity': both_error_sub.sum() / (error1_sub | error2_sub).sum() if (error1_sub | error2_sub).sum() > 0 else 0,
                    'error_correlation': np.corrcoef(subset[f'{model1}_error'], subset[f'{model2}_error'])[0, 1]
                }

        return results

    def get_error_samples(
        self,
        df: pd.DataFrame,
        model_name: str = None,
        error_type: str = 'all'
    ) -> pd.DataFrame:
        """
        Get specific error samples.

        Args:
            df: DataFrame from compute_error_sets()
            model_name: Model name (if error_type needs it)
            error_type: Type of errors to get:
                - 'all': All samples
                - 'model_only': Only this model got it wrong
                - 'model_error': This model got it wrong (regardless of others)
                - 'both_error': Both models got it wrong (requires 2 models in df)

        Returns:
            Filtered DataFrame
        """
        if error_type == 'all':
            return df

        elif error_type == 'model_error':
            return df[df[f'{model_name}_error'] == 1]

        elif error_type == 'model_only':
            # Find the other model
            error_cols = [col for col in df.columns if col.endswith('_error')]
            other_models = [col.replace('_error', '') for col in error_cols if col != f'{model_name}_error']

            if len(other_models) == 0:
                return df[df[f'{model_name}_error'] == 1]

            # This model error AND all others correct
            mask = df[f'{model_name}_error'] == 1
            for other in other_models:
                mask &= df[f'{other}_error'] == 0

            return df[mask]

        elif error_type == 'both_error':
            error_cols = [col for col in df.columns if col.endswith('_error')]
            if len(error_cols) < 2:
                raise ValueError("Need at least 2 models for 'both_error' type")

            mask = df[error_cols[0]] == 1
            for col in error_cols[1:]:
                mask &= df[col] == 1

            return df[mask]

        else:
            raise ValueError(f"Unknown error_type: {error_type}")

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to JSON."""
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Analyze error patterns between models."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze error patterns between models")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vit_b16_quick_test.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--vit-checkpoint',
        type=str,
        default='experiments/vit_b16/quick_test/checkpoints/best_model.pth',
        help='Path to ViT checkpoint'
    )
    parser.add_argument(
        '--resnet-checkpoint',
        type=str,
        default='experiments/resnet50/quick_test/checkpoints/best_model.pth',
        help='Path to ResNet checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/analysis/error_analysis',
        help='Output directory for analysis results'
    )
    args = parser.parse_args()

    print("="*60)
    print("Error Pattern Analysis")
    print("="*60)

    # Initialize analyzer
    analyzer = ErrorAnalyzer(args.config)

    # Load predictions from both models
    model_results = {}

    vit_checkpoint = Path(args.vit_checkpoint)
    if vit_checkpoint.exists() or (project_root / vit_checkpoint).exists():
        print("\n1. Loading ViT predictions...")
        model_results['vit'] = analyzer.load_predictions(args.vit_checkpoint, 'vit_b16')
    else:
        print(f"\nWarning: ViT checkpoint not found at {args.vit_checkpoint}")

    resnet_checkpoint = Path(args.resnet_checkpoint)
    if resnet_checkpoint.exists() or (project_root / resnet_checkpoint).exists():
        print("\n2. Loading ResNet predictions...")
        model_results['resnet'] = analyzer.load_predictions(args.resnet_checkpoint, 'resnet50')
    else:
        print(f"\nWarning: ResNet checkpoint not found at {args.resnet_checkpoint}")

    if len(model_results) == 0:
        print("\nError: No valid checkpoints found!")
        return

    # Compute error sets
    print("\n3. Computing error sets...")
    df = analyzer.compute_error_sets(model_results)

    # Save detailed error data
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'error_samples.csv', index=False)
    print(f"   Saved error samples to: {output_dir / 'error_samples.csv'}")

    # Analyze overlap if we have 2 models
    if len(model_results) >= 2:
        print("\n4. Analyzing error overlap...")
        model1, model2 = list(model_results.keys())[:2]
        overlap_results = analyzer.analyze_error_overlap(df, model1, model2)

        # Print overall results
        print("\n" + "="*60)
        print(f"Overall Error Overlap ({model1} vs {model2})")
        print("="*60)
        overall = overlap_results['overall']
        print(f"Total samples:        {overall['total_samples']}")
        print(f"{model1.upper()} errors:         {overall[f'{model1}_errors']} ({overall[f'{model1}_errors']/overall['total_samples']*100:.1f}%)")
        print(f"{model2.upper()} errors:      {overall[f'{model2}_errors']} ({overall[f'{model2}_errors']/overall['total_samples']*100:.1f}%)")
        print(f"\nBoth correct:         {overall['both_correct']} ({overall['both_correct']/overall['total_samples']*100:.1f}%)")
        print(f"Only {model1.upper()} error:      {overall[f'only_{model1}_error']} ({overall[f'only_{model1}_error']/overall['total_samples']*100:.1f}%)")
        print(f"Only {model2.upper()} error:   {overall[f'only_{model2}_error']} ({overall[f'only_{model2}_error']/overall['total_samples']*100:.1f}%)")
        print(f"Both error:           {overall['both_error']} ({overall['both_error']/overall['total_samples']*100:.1f}%)")
        print(f"\nJaccard similarity:   {overall['jaccard_similarity']:.4f}")
        print(f"Error correlation:    {overall['error_correlation']:.4f}")

        # Print by occlusion
        print("\n" + "="*60)
        print("Error Overlap by Occlusion Level")
        print("="*60)
        for occ_level, stats in overlap_results['by_occlusion'].items():
            print(f"\n{occ_level} Occlusion:")
            print(f"  Both correct:      {stats['both_correct']:3d} ({stats['both_correct']/stats['total_samples']*100:.1f}%)")
            print(f"  Only {model1.upper()} error:   {stats[f'only_{model1}_error']:3d} ({stats[f'only_{model1}_error']/stats['total_samples']*100:.1f}%)")
            print(f"  Only {model2.upper()} error:{stats[f'only_{model2}_error']:3d} ({stats[f'only_{model2}_error']/stats['total_samples']*100:.1f}%)")
            print(f"  Both error:        {stats['both_error']:3d} ({stats['both_error']/stats['total_samples']*100:.1f}%)")
            print(f"  Jaccard similarity: {stats['jaccard_similarity']:.4f}")

        # Save overlap results
        analyzer.save_results(overlap_results, output_dir / 'error_overlap.json')

        # Save error type samples
        print("\n5. Saving error type samples...")
        for error_type in ['vit_only', 'resnet_only', 'both_error']:
            if error_type == 'vit_only':
                samples = analyzer.get_error_samples(df, 'vit', 'model_only')
            elif error_type == 'resnet_only':
                samples = analyzer.get_error_samples(df, 'resnet', 'model_only')
            else:
                samples = analyzer.get_error_samples(df, error_type='both_error')

            samples.to_csv(output_dir / f'{error_type}_errors.csv', index=False)
            print(f"   {error_type}: {len(samples)} samples")

    print("\n" + "="*60)
    print("Error analysis completed!")
    print("="*60)


if __name__ == "__main__":
    main()
