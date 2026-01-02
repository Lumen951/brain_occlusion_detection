"""Compare performance across multiple models and visualize results."""
import sys
from pathlib import Path

# Setup project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


class PerformanceComparator:
    """Compare and visualize performance across models."""

    def __init__(self, output_dir: str = 'experiments/analysis/comparison'):
        """
        Initialize performance comparator.

        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def load_model_results(self, results_path: str) -> pd.DataFrame:
        """
        Load model performance results.

        Args:
            results_path: Path to performance_by_occlusion.csv

        Returns:
            DataFrame with occlusion-level performance
        """
        results_path = Path(results_path)
        if not results_path.is_absolute():
            results_path = project_root / results_path

        return pd.read_csv(results_path)

    def load_human_results(self, results_path: str) -> pd.DataFrame:
        """
        Load human performance results.

        Args:
            results_path: Path to human_performance_by_occlusion.csv

        Returns:
            DataFrame with human occlusion-level performance
        """
        results_path = Path(results_path)
        if not results_path.is_absolute():
            results_path = project_root / results_path

        df = pd.read_csv(results_path)

        # Standardize column names
        df = df.rename(columns={
            'mean_accuracy': 'accuracy',
            'std_accuracy': 'std',
        })

        # Convert occlusion_level to percentage string if needed
        if 'occlusion_level' in df.columns:
            df['occlusion_level'] = df['occlusion_level'].apply(
                lambda x: f"{int(x*100)}%" if isinstance(x, float) else x
            )

        return df

    def create_comparison_dataframe(
        self,
        model_results: Dict[str, pd.DataFrame],
        human_results: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Create unified comparison dataframe.

        Args:
            model_results: Dict mapping model_name -> performance DataFrame
            human_results: Human performance DataFrame (optional)

        Returns:
            Combined DataFrame with all results
        """
        all_data = []

        # Add model results
        for model_name, df in model_results.items():
            for _, row in df.iterrows():
                all_data.append({
                    'model': model_name,
                    'occlusion_level': row['occlusion_level'],
                    'accuracy': row['accuracy'],
                    'std': row.get('std', 0),
                    'type': 'model'
                })

        # Add human results
        if human_results is not None:
            for _, row in human_results.iterrows():
                all_data.append({
                    'model': 'Human',
                    'occlusion_level': row['occlusion_level'],
                    'accuracy': row['accuracy'],
                    'std': row.get('std', 0),
                    'type': 'human'
                })

        return pd.DataFrame(all_data)

    def plot_accuracy_comparison(
        self,
        df: pd.DataFrame,
        title: str = "Model Performance Comparison by Occlusion Level",
        save_name: str = "accuracy_comparison.png"
    ):
        """
        Plot accuracy comparison across models.

        Args:
            df: DataFrame from create_comparison_dataframe()
            title: Plot title
            save_name: Output filename
        """
        plt.figure(figsize=(12, 7))

        # Define occlusion order
        occlusion_order = ['10%', '70%', '90%']

        # Get unique models
        models = df['model'].unique()

        # Color palette
        colors = sns.color_palette("husl", len(models))
        color_map = {model: colors[i] for i, model in enumerate(models)}

        # Use seaborn deprecated warning fix
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)

        # Plot each model
        for model in models:
            model_data = df[df['model'] == model].copy()

            # Sort by occlusion level
            model_data['occ_order'] = model_data['occlusion_level'].map(
                {level: i for i, level in enumerate(occlusion_order)}
            )
            model_data = model_data.sort_values('occ_order')

            # Plot line
            plt.plot(
                model_data['occlusion_level'],
                model_data['accuracy'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=model,
                color=color_map[model]
            )

            # Add error bars if std available
            if model_data['std'].sum() > 0:
                plt.errorbar(
                    model_data['occlusion_level'],
                    model_data['accuracy'],
                    yerr=model_data['std'],
                    fmt='none',
                    ecolor=color_map[model],
                    alpha=0.3,
                    capsize=5
                )

        plt.xlabel('Occlusion Level', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        # Save
        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close()

    def plot_robustness_gap(
        self,
        df: pd.DataFrame,
        save_name: str = "robustness_gap.png"
    ):
        """
        Plot robustness gap (10% accuracy - 90% accuracy).

        Args:
            df: DataFrame from create_comparison_dataframe()
            save_name: Output filename
        """
        # Calculate robustness gap for each model
        gaps = []
        for model in df['model'].unique():
            model_data = df[df['model'] == model]

            acc_10 = model_data[model_data['occlusion_level'] == '10%']['accuracy'].values
            acc_90 = model_data[model_data['occlusion_level'] == '90%']['accuracy'].values

            if len(acc_10) > 0 and len(acc_90) > 0:
                gap = acc_10[0] - acc_90[0]
                gaps.append({
                    'model': model,
                    'robustness_gap': gap,
                    'relative_drop': gap / acc_10[0] * 100
                })

        gaps_df = pd.DataFrame(gaps)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Absolute gap
        sns.barplot(
            data=gaps_df,
            x='model',
            y='robustness_gap',
            hue='model',
            ax=ax1,
            palette='viridis',
            legend=False
        )
        ax1.set_title('Robustness Gap (10% - 90% Accuracy)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Model', fontsize=11)
        ax1.set_ylabel('Accuracy Drop', fontsize=11)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Relative drop
        sns.barplot(
            data=gaps_df,
            x='model',
            y='relative_drop',
            hue='model',
            ax=ax2,
            palette='viridis',
            legend=False
        )
        ax2.set_title('Relative Performance Drop', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Model', fontsize=11)
        ax2.set_ylabel('Percentage Drop (%)', fontsize=11)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Save
        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close()

    def plot_model_human_gap(
        self,
        df: pd.DataFrame,
        save_name: str = "model_human_gap.png"
    ):
        """
        Plot performance gap between models and humans.

        Args:
            df: DataFrame from create_comparison_dataframe()
            save_name: Output filename
        """
        # Get human performance
        human_data = df[df['model'] == 'Human']
        if len(human_data) == 0:
            print("No human data available, skipping model-human gap plot")
            return

        # Calculate gaps for each model
        gaps_by_occ = []
        models = [m for m in df['model'].unique() if m != 'Human']

        if len(models) == 0:
            print("No model data available, skipping model-human gap plot")
            return

        for model in models:
            model_data = df[df['model'] == model]

            for occ_level in ['10%', '70%', '90%']:
                model_acc = model_data[model_data['occlusion_level'] == occ_level]['accuracy'].values
                human_acc = human_data[human_data['occlusion_level'] == occ_level]['accuracy'].values

                if len(model_acc) > 0 and len(human_acc) > 0:
                    gaps_by_occ.append({
                        'model': model,
                        'occlusion_level': occ_level,
                        'gap': model_acc[0] - human_acc[0]
                    })

        if len(gaps_by_occ) == 0:
            print("No overlapping occlusion levels found, skipping model-human gap plot")
            return

        gaps_df = pd.DataFrame(gaps_by_occ)

        # Plot grouped bar chart
        plt.figure(figsize=(12, 7))

        x = np.arange(len(['10%', '70%', '90%']))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_gaps = gaps_df[gaps_df['model'] == model].sort_values('occlusion_level')

            # Get gaps in correct order
            gap_values = []
            for occ in ['10%', '70%', '90%']:
                gap_row = model_gaps[model_gaps['occlusion_level'] == occ]
                if len(gap_row) > 0:
                    gap_values.append(gap_row['gap'].values[0])
                else:
                    gap_values.append(0)

            plt.bar(
                x + i * width,
                gap_values,
                width,
                label=model,
                alpha=0.8
            )

        plt.xlabel('Occlusion Level', fontsize=12)
        plt.ylabel('Accuracy Gap (Model - Human)', fontsize=12)
        plt.title('Model vs Human Performance Gap', fontsize=14, fontweight='bold')
        plt.xticks(x + width * (len(models) - 1) / 2, ['10%', '70%', '90%'])
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')

        # Save
        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close()

    def generate_summary_table(
        self,
        df: pd.DataFrame,
        save_name: str = "performance_summary.csv"
    ) -> pd.DataFrame:
        """
        Generate summary table with key metrics.

        Args:
            df: DataFrame from create_comparison_dataframe()
            save_name: Output filename

        Returns:
            Summary DataFrame
        """
        summary = []

        for model in df['model'].unique():
            model_data = df[df['model'] == model]

            # Overall accuracy
            overall_acc = model_data['accuracy'].mean()

            # Per-occlusion accuracy
            acc_10 = model_data[model_data['occlusion_level'] == '10%']['accuracy'].values
            acc_70 = model_data[model_data['occlusion_level'] == '70%']['accuracy'].values
            acc_90 = model_data[model_data['occlusion_level'] == '90%']['accuracy'].values

            # Robustness metrics
            if len(acc_10) > 0 and len(acc_90) > 0:
                robustness_gap = acc_10[0] - acc_90[0]
                relative_drop = robustness_gap / acc_10[0] * 100
            else:
                robustness_gap = np.nan
                relative_drop = np.nan

            summary.append({
                'Model': model,
                'Overall Accuracy': overall_acc,
                '10% Accuracy': acc_10[0] if len(acc_10) > 0 else np.nan,
                '70% Accuracy': acc_70[0] if len(acc_70) > 0 else np.nan,
                '90% Accuracy': acc_90[0] if len(acc_90) > 0 else np.nan,
                'Robustness Gap': robustness_gap,
                'Relative Drop (%)': relative_drop
            })

        summary_df = pd.DataFrame(summary)

        # Save
        save_path = self.output_dir / save_name
        summary_df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"Saved summary: {save_path}")

        return summary_df


def main():
    """Generate performance comparison visualizations."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument(
        '--vit-results',
        type=str,
        default='scripts/experiments/vit_b16/quick_test/metrics/performance_by_occlusion.csv',
        help='Path to ViT results CSV'
    )
    parser.add_argument(
        '--resnet-results',
        type=str,
        default='scripts/experiments/resnet50/quick_test/metrics/performance_by_occlusion.csv',
        help='Path to ResNet results CSV'
    )
    parser.add_argument(
        '--human-results',
        type=str,
        default='data/human_performance/human_performance_by_occlusion.csv',
        help='Path to human results CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scripts/experiments/analysis/comparison',
        help='Output directory for plots'
    )
    args = parser.parse_args()

    print("="*60)
    print("Performance Comparison")
    print("="*60)

    # Initialize comparator
    comparator = PerformanceComparator(args.output_dir)

    # Load model results
    model_results = {}

    vit_path = Path(args.vit_results)
    if vit_path.exists() or (project_root / vit_path).exists():
        print("\n1. Loading ViT results...")
        model_results['ViT-B/16'] = comparator.load_model_results(args.vit_results)
        print(f"   Loaded {len(model_results['ViT-B/16'])} occlusion levels")
    else:
        print(f"\nWarning: ViT results not found at {args.vit_results}")

    resnet_path = Path(args.resnet_results)
    if resnet_path.exists() or (project_root / resnet_path).exists():
        print("\n2. Loading ResNet results...")
        model_results['ResNet-50'] = comparator.load_model_results(args.resnet_results)
        print(f"   Loaded {len(model_results['ResNet-50'])} occlusion levels")
    else:
        print(f"\nWarning: ResNet results not found at {args.resnet_results}")

    # Load human results
    human_results = None
    human_path = Path(args.human_results)
    if human_path.exists() or (project_root / human_path).exists():
        print("\n3. Loading human results...")
        human_results = comparator.load_human_results(args.human_results)
        print(f"   Loaded {len(human_results)} occlusion levels")
    else:
        print(f"\nWarning: Human results not found at {args.human_results}")

    if len(model_results) == 0:
        print("\nError: No model results found!")
        return

    # Create comparison dataframe
    print("\n4. Creating comparison dataframe...")
    comparison_df = comparator.create_comparison_dataframe(model_results, human_results)

    # Generate plots
    print("\n5. Generating plots...")
    comparator.plot_accuracy_comparison(comparison_df)
    comparator.plot_robustness_gap(comparison_df)

    if human_results is not None:
        comparator.plot_model_human_gap(comparison_df)

    # Generate summary table
    print("\n6. Generating summary table...")
    summary_df = comparator.generate_summary_table(comparison_df)

    # Print summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(summary_df.to_string(index=False))

    print("\n" + "="*60)
    print("Comparison completed!")
    print(f"Results saved to: {comparator.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
