"""Generate all possible comparison visualizations between models."""
import sys
from pathlib import Path

# Setup project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional


class AllComparisonsVisualizer:
    """Generate comprehensive visualizations for model comparison."""

    def __init__(self, output_dir: str = 'experiments/analysis/all_visualizations'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10

    def plot_training_loss_comparison(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "01_loss_comparison.png"
    ):
        """Training and validation loss side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Loss Comparison', fontsize=14, fontweight='bold')

        colors = {'ViT-B/16': '#1f77b4', 'ResNet-50': '#ff7f0e'}

        # Training loss
        for model, df in histories.items():
            ax1.plot(df['epoch'], df['train_loss'],
                    label=model, color=colors[model],
                    marker='o', markersize=4, linewidth=2, alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation loss
        for model, df in histories.items():
            ax2.plot(df['epoch'], df['val_loss'],
                    label=model, color=colors[model],
                    marker='s', markersize=4, linewidth=2, alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_accuracy_progression(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "02_accuracy_progression.png"
    ):
        """Training and validation accuracy progression."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        colors = {'ViT-B/16': '#1f77b4', 'ResNet-50': '#ff7f0e'}

        for model, df in histories.items():
            # Training accuracy (dashed)
            ax.plot(df['epoch'], df['train_acc'] * 100,
                   label=f'{model} Train', color=colors[model],
                   linestyle='--', linewidth=2, alpha=0.6)

            # Validation accuracy (solid)
            ax.plot(df['epoch'], df['val_acc'] * 100,
                   label=f'{model} Val', color=colors[model],
                   linestyle='-', linewidth=2.5, marker='o', markersize=5)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy Progression (Solid=Val, Dashed=Train)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_f1_comparison(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "03_f1_comparison.png"
    ):
        """F1 score comparison."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        colors = {'ViT-B/16': '#1f77b4', 'ResNet-50': '#ff7f0e'}

        for model, df in histories.items():
            ax.plot(df['epoch'], df['val_f1'] * 100,
                   label=model, color=colors[model],
                   marker='D', markersize=5, linewidth=2.5, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1 Score (%)', fontsize=12)
        ax.set_title('Validation F1 Score Comparison',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_overfitting_analysis(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "04_overfitting_analysis.png"
    ):
        """Train-val gap to detect overfitting."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        colors = {'ViT-B/16': '#1f77b4', 'ResNet-50': '#ff7f0e'}

        for model, df in histories.items():
            gap = (df['train_acc'] - df['val_acc']) * 100
            ax.plot(df['epoch'], gap,
                   label=model, color=colors[model],
                   marker='o', markersize=5, linewidth=2, alpha=0.7)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=10, color='red', linestyle=':', alpha=0.5,
                  label='Overfitting threshold (10%)')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train-Val Accuracy Gap (%)', fontsize=12)
        ax.set_title('Overfitting Analysis (Higher = More Overfitting)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_learning_rate_schedule(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "05_learning_rate.png"
    ):
        """Learning rate schedule comparison."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        colors = {'ViT-B/16': '#1f77b4', 'ResNet-50': '#ff7f0e'}

        for model, df in histories.items():
            ax.plot(df['epoch'], df['learning_rate'],
                   label=model, color=colors[model],
                   marker='o', markersize=4, linewidth=2, alpha=0.7)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_metric_heatmap(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "06_metrics_heatmap.png"
    ):
        """Heatmap of all metrics over time."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Metrics Evolution Heatmap', fontsize=14, fontweight='bold')

        for idx, (model, df) in enumerate(histories.items()):
            # Select key metrics
            metrics_df = df[['epoch', 'train_loss', 'val_loss',
                           'train_acc', 'val_acc', 'val_f1']].copy()

            # Normalize for heatmap (0-1 scale)
            normalized = metrics_df.copy()
            for col in ['train_loss', 'val_loss']:
                normalized[col] = 1 - (normalized[col] - normalized[col].min()) / \
                                     (normalized[col].max() - normalized[col].min() + 1e-8)

            # Create heatmap data
            heatmap_data = normalized.drop('epoch', axis=1).T

            sns.heatmap(heatmap_data, ax=axes[idx],
                       cmap='RdYlGn', center=0.5,
                       xticklabels=df['epoch'].astype(int),
                       yticklabels=['Train Loss↓', 'Val Loss↓',
                                   'Train Acc', 'Val Acc', 'Val F1'],
                       cbar_kws={'label': 'Normalized Score'})

            axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch')

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_final_comparison_bars(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "07_final_comparison.png"
    ):
        """Bar chart comparing final and best metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Final vs Best Metrics Comparison',
                    fontsize=14, fontweight='bold')

        models = list(histories.keys())
        x = np.arange(len(models))
        width = 0.35

        # Extract metrics
        best_val_acc = [histories[m]['val_acc'].max() * 100 for m in models]
        final_val_acc = [histories[m]['val_acc'].iloc[-1] * 100 for m in models]
        best_val_f1 = [histories[m]['val_f1'].max() * 100 for m in models]
        final_val_f1 = [histories[m]['val_f1'].iloc[-1] * 100 for m in models]
        min_val_loss = [histories[m]['val_loss'].min() for m in models]
        final_val_loss = [histories[m]['val_loss'].iloc[-1] for m in models]

        # 1. Validation Accuracy
        ax = axes[0, 0]
        ax.bar(x - width/2, best_val_acc, width, label='Best', alpha=0.8)
        ax.bar(x + width/2, final_val_acc, width, label='Final', alpha=0.8)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Validation Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        # Add values on bars
        for i, (best, final) in enumerate(zip(best_val_acc, final_val_acc)):
            ax.text(i - width/2, best, f'{best:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, final, f'{final:.1f}', ha='center', va='bottom', fontsize=9)

        # 2. F1 Score
        ax = axes[0, 1]
        ax.bar(x - width/2, best_val_f1, width, label='Best', alpha=0.8)
        ax.bar(x + width/2, final_val_f1, width, label='Final', alpha=0.8)
        ax.set_ylabel('F1 Score (%)')
        ax.set_title('Validation F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for i, (best, final) in enumerate(zip(best_val_f1, final_val_f1)):
            ax.text(i - width/2, best, f'{best:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, final, f'{final:.1f}', ha='center', va='bottom', fontsize=9)

        # 3. Validation Loss
        ax = axes[1, 0]
        ax.bar(x - width/2, min_val_loss, width, label='Min', alpha=0.8)
        ax.bar(x + width/2, final_val_loss, width, label='Final', alpha=0.8)
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        for i, (min_l, final_l) in enumerate(zip(min_val_loss, final_val_loss)):
            ax.text(i - width/2, min_l, f'{min_l:.4f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, final_l, f'{final_l:.4f}', ha='center', va='bottom', fontsize=8)

        # 4. Epochs trained
        ax = axes[1, 1]
        epochs = [len(histories[m]) for m in models]
        bars = ax.bar(models, epochs, alpha=0.8)
        ax.set_ylabel('Number of Epochs')
        ax.set_title('Training Duration')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, epoch in zip(bars, epochs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{epoch}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_epoch_by_epoch_delta(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "08_performance_delta.png"
    ):
        """Show epoch-by-epoch performance difference."""
        if len(histories) != 2:
            print(f"⊘ Skipping {save_name} (requires exactly 2 models)")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('ViT-B/16 vs ResNet-50 Performance Delta',
                    fontsize=14, fontweight='bold')

        models = list(histories.keys())
        df1, df2 = histories[models[0]], histories[models[1]]

        # Align to same number of epochs
        min_epochs = min(len(df1), len(df2))
        df1 = df1.iloc[:min_epochs]
        df2 = df2.iloc[:min_epochs]

        epochs = df1['epoch']

        # 1. Validation accuracy delta
        ax = axes[0]
        delta_acc = (df1['val_acc'] - df2['val_acc']) * 100
        colors = ['green' if x > 0 else 'red' for x in delta_acc]
        ax.bar(epochs, delta_acc, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{models[0]} - {models[1]} (%)')
        ax.set_title('Validation Accuracy Difference (Positive = ViT Better)')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Validation F1 delta
        ax = axes[1]
        delta_f1 = (df1['val_f1'] - df2['val_f1']) * 100
        colors = ['green' if x > 0 else 'red' for x in delta_f1]
        ax.bar(epochs, delta_f1, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{models[0]} - {models[1]} (%)')
        ax.set_title('Validation F1 Difference (Positive = ViT Better)')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def plot_summary_table(
        self,
        histories: Dict[str, pd.DataFrame],
        save_name: str = "09_summary_table.png"
    ):
        """Generate a summary table as an image."""
        # Collect data
        summary_data = []
        for model, df in histories.items():
            summary_data.append({
                'Model': model,
                'Best Val Acc': f"{df['val_acc'].max()*100:.2f}%",
                'Best Val F1': f"{df['val_f1'].max()*100:.2f}%",
                'Min Val Loss': f"{df['val_loss'].min():.4f}",
                'Final Val Acc': f"{df['val_acc'].iloc[-1]*100:.2f}%",
                'Final Val F1': f"{df['val_f1'].iloc[-1]*100:.2f}%",
                'Epochs': len(df)
            })

        summary_df = pd.DataFrame(summary_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.13, 0.13, 0.13, 0.15, 0.13, 0.08])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(summary_data) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / save_name, bbox_inches='tight')
        print(f"✓ {save_name}")
        plt.close()

    def generate_all_visualizations(self, histories: Dict[str, pd.DataFrame]):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("Generating All Comparison Visualizations")
        print("="*60 + "\n")

        self.plot_training_loss_comparison(histories)
        self.plot_accuracy_progression(histories)
        self.plot_f1_comparison(histories)
        self.plot_overfitting_analysis(histories)
        self.plot_learning_rate_schedule(histories)
        self.plot_metric_heatmap(histories)
        self.plot_final_comparison_bars(histories)
        self.plot_epoch_by_epoch_delta(histories)
        self.plot_summary_table(histories)

        print("\n" + "="*60)
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate all comparison visualizations")
    parser.add_argument(
        '--vit-history',
        type=str,
        default='experiments/vit_b16/image_split/training_history.csv',
        help='Path to ViT training history'
    )
    parser.add_argument(
        '--resnet-history',
        type=str,
        default='experiments/resnet50/image_split/training_history.csv',
        help='Path to ResNet training history'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/analysis/all_visualizations',
        help='Output directory'
    )
    args = parser.parse_args()

    # Initialize
    visualizer = AllComparisonsVisualizer(args.output_dir)

    # Load histories
    histories = {}

    vit_path = Path(args.vit_history)
    if not vit_path.is_absolute():
        vit_path = project_root / vit_path

    if vit_path.exists():
        histories['ViT-B/16'] = pd.read_csv(vit_path)
        print(f"✓ Loaded ViT-B/16: {len(histories['ViT-B/16'])} epochs")
    else:
        print(f"✗ ViT history not found: {vit_path}")

    resnet_path = Path(args.resnet_history)
    if not resnet_path.is_absolute():
        resnet_path = project_root / resnet_path

    if resnet_path.exists():
        histories['ResNet-50'] = pd.read_csv(resnet_path)
        print(f"✓ Loaded ResNet-50: {len(histories['ResNet-50'])} epochs")
    else:
        print(f"✗ ResNet history not found: {resnet_path}")

    if len(histories) == 0:
        print("\n✗ No training histories found!")
        return

    # Generate all visualizations
    visualizer.generate_all_visualizations(histories)


if __name__ == "__main__":
    main()
