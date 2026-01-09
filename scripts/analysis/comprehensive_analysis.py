"""
Comprehensive Analysis Script for Brain Occlusion Detection Project
Generates statistical analysis and visualizations for research report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Project root
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "experiments" / "analysis"
output_dir = project_root / "reports" / "analysis_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all experimental data"""
    print("Loading experimental data...")

    # Load metrics by occlusion
    resnet_metrics = pd.read_csv(data_dir / "resnet50_metrics_by_occlusion.csv")
    vit_metrics = pd.read_csv(data_dir / "vit_b16_metrics_by_occlusion.csv")

    # Load per-image comparison
    per_image = pd.read_csv(data_dir / "per_image_comparison_summary.csv")

    # Load training history
    resnet_history = pd.read_csv(project_root / "experiments" / "resnet50" / "image_split" / "training_history.csv")
    vit_history = pd.read_csv(project_root / "experiments" / "vit_b16" / "image_split" / "training_history.csv")

    return {
        'resnet_metrics': resnet_metrics,
        'vit_metrics': vit_metrics,
        'per_image': per_image,
        'resnet_history': resnet_history,
        'vit_history': vit_history
    }

def statistical_analysis(data):
    """Perform statistical tests"""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    results = {}

    # 1. Compare ViT vs ResNet accuracy across occlusion levels
    print("\n1. Model Performance Comparison (ViT vs ResNet)")
    print("-" * 60)

    for idx, row in data['vit_metrics'].iterrows():
        occlusion = row['occlusion_level']
        vit_acc = row['accuracy']
        resnet_acc = data['resnet_metrics'].iloc[idx]['accuracy']

        print(f"\nOcclusion {occlusion}:")
        print(f"  ViT Accuracy:    {vit_acc:.2%}")
        print(f"  ResNet Accuracy: {resnet_acc:.2%}")
        print(f"  Difference:      {(vit_acc - resnet_acc):.2%}")

    # 2. Human vs AI gap analysis
    print("\n\n2. Human vs AI Performance Gap")
    print("-" * 60)

    # Human performance from existing report
    human_perf = {
        '10%': 0.9562,
        '70%': 0.7928,
        '90%': 0.6188
    }

    gaps = []
    for idx, row in data['vit_metrics'].iterrows():
        occlusion = row['occlusion_level']
        vit_acc = row['accuracy']
        resnet_acc = data['resnet_metrics'].iloc[idx]['accuracy']
        human_acc = human_perf[occlusion]

        vit_gap = human_acc - vit_acc
        resnet_gap = human_acc - resnet_acc

        print(f"\nOcclusion {occlusion}:")
        print(f"  Human:       {human_acc:.2%}")
        print(f"  ViT Gap:     {vit_gap:.2%}")
        print(f"  ResNet Gap:  {resnet_gap:.2%}")

        gaps.append({
            'occlusion': occlusion,
            'human': human_acc,
            'vit': vit_acc,
            'resnet': resnet_acc,
            'vit_gap': vit_gap,
            'resnet_gap': resnet_gap
        })

    results['gaps'] = pd.DataFrame(gaps)

    # 3. Agreement analysis
    print("\n\n3. Human-AI Agreement Analysis")
    print("-" * 60)

    per_image = data['per_image']

    # Calculate agreement metrics
    all_correct = per_image['all_correct'].sum()
    all_wrong = per_image['all_wrong'].sum()

    # Human correct, both AI wrong
    human_only = ((per_image['Human'] == 1) &
                  (per_image['ViT'] == 0) &
                  (per_image['ResNet'] == 0)).sum()

    # AI correct, human wrong
    ai_only = ((per_image['Human'] == 0) &
               ((per_image['ViT'] == 1) | (per_image['ResNet'] == 1))).sum()

    print(f"Total test images: {len(per_image)}")
    print(f"All three correct: {all_correct} ({all_correct/len(per_image):.1%})")
    print(f"All three wrong:   {all_wrong} ({all_wrong/len(per_image):.1%})")
    print(f"Only human correct: {human_only} ({human_only/len(per_image):.1%})")
    print(f"Only AI correct:    {ai_only} ({ai_only/len(per_image):.1%})")

    # Agreement by occlusion level
    print("\nAgreement by occlusion level:")
    for occlusion in ['10%', '70%', '90%']:
        subset = per_image[per_image['Occlusion'] == occlusion]
        all_correct_occ = subset['all_correct'].sum()
        print(f"  {occlusion}: {all_correct_occ}/{len(subset)} all correct ({all_correct_occ/len(subset):.1%})")

    results['agreement'] = {
        'all_correct': all_correct,
        'all_wrong': all_wrong,
        'human_only': human_only,
        'ai_only': ai_only
    }

    # 4. Training convergence analysis
    print("\n\n4. Training Convergence Analysis")
    print("-" * 60)

    for model_name, history in [('ViT', data['vit_history']), ('ResNet', data['resnet_history'])]:
        best_epoch = history['val_acc'].idxmax() + 1
        best_val_acc = history['val_acc'].max()
        final_train_acc = history.iloc[best_epoch-1]['train_acc']

        overfitting_gap = final_train_acc - best_val_acc

        print(f"\n{model_name}:")
        print(f"  Best epoch:        {best_epoch}")
        print(f"  Best val accuracy: {best_val_acc:.2%}")
        print(f"  Train accuracy:    {final_train_acc:.2%}")
        print(f"  Overfitting gap:   {overfitting_gap:.2%}")

    return results

def create_visualizations(data, stats_results):
    """Create comprehensive visualizations"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Human vs AI Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    gaps_df = stats_results['gaps']
    x = np.arange(len(gaps_df))
    width = 0.25

    ax.bar(x - width, gaps_df['human'], width, label='Human', color='#2ecc71')
    ax.bar(x, gaps_df['vit'], width, label='ViT-B/16', color='#3498db')
    ax.bar(x + width, gaps_df['resnet'], width, label='ResNet-50', color='#e74c3c')

    ax.set_xlabel('Occlusion Level', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Human vs AI Performance Across Occlusion Levels', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gaps_df['occlusion'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'human_vs_ai_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: human_vs_ai_comparison.png")
    plt.close()

    # 2. Performance Gap Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    gap_matrix = gaps_df[['vit_gap', 'resnet_gap']].T
    gap_matrix.columns = gaps_df['occlusion']
    gap_matrix.index = ['ViT Gap', 'ResNet Gap']

    sns.heatmap(gap_matrix, annot=True, fmt='.2%', cmap='RdYlGn_r',
                cbar_kws={'label': 'Performance Gap'}, ax=ax)
    ax.set_title('Human-AI Performance Gap Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_gap_heatmap.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: performance_gap_heatmap.png")
    plt.close()

    # 3. Training Dynamics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(data['vit_history']['epoch'], data['vit_history']['train_loss'],
                    label='ViT Train', color='#3498db', linewidth=2)
    axes[0, 0].plot(data['vit_history']['epoch'], data['vit_history']['val_loss'],
                    label='ViT Val', color='#3498db', linestyle='--', linewidth=2)
    axes[0, 0].plot(data['resnet_history']['epoch'], data['resnet_history']['train_loss'],
                    label='ResNet Train', color='#e74c3c', linewidth=2)
    axes[0, 0].plot(data['resnet_history']['epoch'], data['resnet_history']['val_loss'],
                    label='ResNet Val', color='#e74c3c', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(data['vit_history']['epoch'], data['vit_history']['train_acc'],
                    label='ViT Train', color='#3498db', linewidth=2)
    axes[0, 1].plot(data['vit_history']['epoch'], data['vit_history']['val_acc'],
                    label='ViT Val', color='#3498db', linestyle='--', linewidth=2)
    axes[0, 1].plot(data['resnet_history']['epoch'], data['resnet_history']['train_acc'],
                    label='ResNet Train', color='#e74c3c', linewidth=2)
    axes[0, 1].plot(data['resnet_history']['epoch'], data['resnet_history']['val_acc'],
                    label='ResNet Val', color='#e74c3c', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # F1 Score curves
    axes[1, 0].plot(data['vit_history']['epoch'], data['vit_history']['val_f1'],
                    label='ViT', color='#3498db', linewidth=2)
    axes[1, 0].plot(data['resnet_history']['epoch'], data['resnet_history']['val_f1'],
                    label='ResNet', color='#e74c3c', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Overfitting analysis
    vit_overfit = data['vit_history']['train_acc'] - data['vit_history']['val_acc']
    resnet_overfit = data['resnet_history']['train_acc'] - data['resnet_history']['val_acc']

    axes[1, 1].plot(data['vit_history']['epoch'], vit_overfit,
                    label='ViT', color='#3498db', linewidth=2)
    axes[1, 1].plot(data['resnet_history']['epoch'], resnet_overfit,
                    label='ResNet', color='#e74c3c', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Acc - Val Acc')
    axes[1, 1].set_title('Overfitting Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: training_dynamics.png")
    plt.close()

    # 4. Agreement Analysis
    fig, ax = plt.subplots(figsize=(8, 6))

    agreement_data = stats_results['agreement']
    categories = ['All Correct', 'All Wrong', 'Human Only', 'AI Only']
    values = [agreement_data['all_correct'], agreement_data['all_wrong'],
              agreement_data['human_only'], agreement_data['ai_only']]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    ax.bar(categories, values, color=colors)
    ax.set_ylabel('Number of Images')
    ax.set_title('Human-AI Agreement Patterns', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'agreement_analysis.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: agreement_analysis.png")
    plt.close()

def generate_summary_report(data, stats_results):
    """Generate text summary report"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)

    report = []
    report.append("# Comprehensive Analysis Summary\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Key Findings\n\n")

    # Finding 1: Performance gaps
    gaps_df = stats_results['gaps']
    avg_vit_gap = gaps_df['vit_gap'].mean()
    avg_resnet_gap = gaps_df['resnet_gap'].mean()

    report.append(f"### 1. Human-AI Performance Gap\n")
    report.append(f"- Average ViT gap: {avg_vit_gap:.2%}\n")
    report.append(f"- Average ResNet gap: {avg_resnet_gap:.2%}\n")
    report.append(f"- ViT performs {(avg_resnet_gap - avg_vit_gap):.2%} better than ResNet on average\n\n")

    # Finding 2: Occlusion effects
    report.append(f"### 2. Occlusion Level Effects\n")
    for idx, row in gaps_df.iterrows():
        report.append(f"- {row['occlusion']}: ViT gap = {row['vit_gap']:.2%}, ResNet gap = {row['resnet_gap']:.2%}\n")
    report.append("\n")

    # Finding 3: Agreement
    agreement = stats_results['agreement']
    total_images = 48
    report.append(f"### 3. Human-AI Agreement\n")
    report.append(f"- All three correct: {agreement['all_correct']}/{total_images} ({agreement['all_correct']/total_images:.1%})\n")
    report.append(f"- Human correct, AI wrong: {agreement['human_only']}/{total_images} ({agreement['human_only']/total_images:.1%})\n")
    report.append(f"- AI correct, human wrong: {agreement['ai_only']}/{total_images} ({agreement['ai_only']/total_images:.1%})\n\n")

    # Save report
    report_text = ''.join(report)
    with open(output_dir / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("[OK] Saved: analysis_summary.txt")
    print("\n" + report_text)

    return report_text

def main():
    print("="*60)
    print("COMPREHENSIVE ANALYSIS FOR BRAIN OCCLUSION DETECTION")
    print("="*60)

    # Load data
    data = load_data()

    # Statistical analysis
    stats_results = statistical_analysis(data)

    # Create visualizations
    create_visualizations(data, stats_results)

    # Generate summary
    summary = generate_summary_report(data, stats_results)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - human_vs_ai_comparison.png")
    print("  - performance_gap_heatmap.png")
    print("  - training_dynamics.png")
    print("  - agreement_analysis.png")
    print("  - analysis_summary.txt")

if __name__ == "__main__":
    main()
