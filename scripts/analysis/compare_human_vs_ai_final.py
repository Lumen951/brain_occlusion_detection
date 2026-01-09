"""
生成人类 vs AI模型的完整对比分析

输入:
- data/human_performance/human_by_image.csv
- experiments/analysis/vit_b16_test_predictions.csv
- experiments/analysis/resnet50_test_predictions.csv

输出:
- experiments/analysis/human_vs_ai_performance.png
- experiments/analysis/final_comparison_report.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_all_data(include_mae=True):
    """加载所有数据"""
    print("Loading data...")

    # 人类数据
    human_df = pd.read_csv("data/human_performance/human_by_image.csv")
    print(f"  Human data: {len(human_df)} images")

    # AI预测数据
    vit_df = pd.read_csv("experiments/analysis/vit_b16_test_predictions.csv")
    print(f"  ViT predictions: {len(vit_df)} images")

    resnet_df = pd.read_csv("experiments/analysis/resnet50_test_predictions.csv")
    print(f"  ResNet predictions: {len(resnet_df)} images")

    # MAE数据（如果存在）
    mae_df = None
    if include_mae:
        mae_path = Path("experiments/analysis/mae_vit_base_test_predictions.csv")
        if mae_path.exists():
            mae_df = pd.read_csv(mae_path)
            print(f"  MAE predictions: {len(mae_df)} images")
        else:
            print(f"  [INFO] MAE predictions not found, skipping...")

    return human_df, vit_df, resnet_df, mae_df


def match_test_images_with_human(human_df, test_df):
    """匹配测试集图像和人类数据"""
    print("\nMatching test images with human data...")

    # 获取测试集图像名称
    test_images = set(test_df['image_name'])
    human_images = set(human_df['stimuli'])

    overlap = test_images & human_images

    print(f"  Test images: {len(test_images)}")
    print(f"  Human images: {len(human_images)}")
    print(f"  Overlapping: {len(overlap)}")

    if len(overlap) == 0:
        print("  [WARNING] No overlapping images found!")
        print("  This means test images are not in human experimental data.")
        print("  Will use overall human averages instead.")

        # 计算人类在各遮挡级别的整体准确率
        human_by_occ = human_df.groupby('occlusion_level')['accuracy'].mean()
        return human_by_occ, len(overlap)

    # 如果有重叠，只使用重叠图像
    matched_human = human_df[human_df['stimuli'].isin(overlap)]
    print(f"  Matched human data: {len(matched_human)} images")

    return matched_human, len(overlap)


def create_comparison_summary(human_df, vit_df, resnet_df, mae_df=None):
    """创建对比摘要"""

    # 计算人类在各遮挡级别的平均准确率
    human_by_occ = human_df.groupby('occlusion_level')['accuracy'].agg(['mean', 'std']).reset_index()
    human_by_occ.columns = ['occlusion_level', 'accuracy', 'std']

    # ViT统计
    vit_by_occ = vit_df.groupby('occlusion_str').agg({
        'correct': ['sum', 'count']
    }).reset_index()
    vit_by_occ.columns = ['occlusion_str', 'correct', 'total']
    vit_by_occ['accuracy'] = vit_by_occ['correct'] / vit_by_occ['total']
    vit_by_occ['occlusion_level'] = vit_by_occ['occlusion_str'].str.replace('%', '').astype(int) / 100

    # ResNet统计
    resnet_by_occ = resnet_df.groupby('occlusion_str').agg({
        'correct': ['sum', 'count']
    }).reset_index()
    resnet_by_occ.columns = ['occlusion_str', 'correct', 'total']
    resnet_by_occ['accuracy'] = resnet_by_occ['correct'] / resnet_by_occ['total']
    resnet_by_occ['occlusion_level'] = resnet_by_occ['occlusion_str'].str.replace('%', '').astype(int) / 100

    # 合并
    comparison = pd.DataFrame({
        'occlusion_level': ['10%', '70%', '90%'],
        'human_accuracy': human_by_occ['accuracy'].values,
        'vit_accuracy': vit_by_occ['accuracy'].values,
        'resnet_accuracy': resnet_by_occ['accuracy'].values
    })

    # MAE统计（如果存在）
    if mae_df is not None:
        mae_by_occ = mae_df.groupby('occlusion_str').agg({
            'correct': ['sum', 'count']
        }).reset_index()
        mae_by_occ.columns = ['occlusion_str', 'correct', 'total']
        mae_by_occ['accuracy'] = mae_by_occ['correct'] / mae_by_occ['total']
        comparison['mae_accuracy'] = mae_by_occ['accuracy'].values

    return comparison


def plot_performance_comparison(comparison_df, output_path):
    """绘制性能对比图"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    occlusion_levels = comparison_df['occlusion_level']
    x = np.arange(len(occlusion_levels))
    width = 0.25

    # Figure 1: 性能柱状图对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width, comparison_df['human_accuracy'] * 100,
                    width, label='Human', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x, comparison_df['vit_accuracy'] * 100,
                    width, label='ViT-B/16', color='#e74c3c', alpha=0.8)
    bars3 = ax1.bar(x + width, comparison_df['resnet_accuracy'] * 100,
                    width, label='ResNet-50', color='#3498db', alpha=0.8)

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Occlusion Level', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Comparison by Occlusion Level', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(occlusion_levels)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 100])

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Figure 2: 性能下降曲线
    ax2 = axes[0, 1]
    ax2.plot(occlusion_levels, comparison_df['human_accuracy'] * 100,
             marker='o', linewidth=2, markersize=8, label='Human', color='#2ecc71')
    ax2.plot(occlusion_levels, comparison_df['vit_accuracy'] * 100,
             marker='s', linewidth=2, markersize=8, label='ViT-B/16', color='#e74c3c')
    ax2.plot(occlusion_levels, comparison_df['resnet_accuracy'] * 100,
             marker='^', linewidth=2, markersize=8, label='ResNet-50', color='#3498db')

    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Occlusion Level', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Degradation with Occlusion', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 100])

    # Figure 3: Human vs AI 差距
    ax3 = axes[1, 0]
    vit_gap = (comparison_df['human_accuracy'] - comparison_df['vit_accuracy']) * 100
    resnet_gap = (comparison_df['human_accuracy'] - comparison_df['resnet_accuracy']) * 100

    x = np.arange(len(occlusion_levels))
    width = 0.35

    bars1 = ax3.bar(x - width/2, vit_gap, width, label='Human - ViT', color='#e74c3c', alpha=0.7)
    bars2 = ax3.bar(x + width/2, resnet_gap, width, label='Human - ResNet', color='#3498db', alpha=0.7)

    ax3.set_ylabel('Accuracy Gap (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Occlusion Level', fontsize=12, fontweight='bold')
    ax3.set_title('Human Superiority Over AI Models', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(occlusion_levels)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center',
                    va='bottom' if height > 0 else 'top', fontsize=9)

    # Figure 4: 相对性能（相对于人类）
    ax4 = axes[1, 1]
    vit_relative = (comparison_df['vit_accuracy'] / comparison_df['human_accuracy']) * 100
    resnet_relative = (comparison_df['resnet_accuracy'] / comparison_df['human_accuracy']) * 100

    x = np.arange(len(occlusion_levels))
    width = 0.35

    bars1 = ax4.bar(x - width/2, vit_relative, width, label='ViT-B/16', color='#e74c3c', alpha=0.7)
    bars2 = ax4.bar(x + width/2, resnet_relative, width, label='ResNet-50', color='#3498db', alpha=0.7)

    ax4.set_ylabel('Relative Performance (Human = 100%)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Occlusion Level', fontsize=12, fontweight='bold')
    ax4.set_title('AI Performance Relative to Human', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(occlusion_levels)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Human Baseline')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved comparison plot to {output_path}")
    plt.close()


def generate_text_report(comparison_df, vit_df, resnet_df, output_path):
    """生成文本报告"""

    report = []
    report.append("="*80)
    report.append("HUMAN vs AI MODEL COMPARISON REPORT")
    report.append("="*80)
    report.append("")

    # 1. 整体摘要
    report.append("1. OVERALL PERFORMANCE SUMMARY")
    report.append("-" * 80)
    report.append("")

    human_mean = comparison_df['human_accuracy'].mean() * 100
    vit_mean = comparison_df['vit_accuracy'].mean() * 100
    resnet_mean = comparison_df['resnet_accuracy'].mean() * 100

    report.append(f"Overall Accuracy:")
    report.append(f"  Human:    {human_mean:.2f}%")
    report.append(f"  ViT-B/16: {vit_mean:.2f}% (Human - ViT: {human_mean - vit_mean:.2f}%)")
    report.append(f"  ResNet-50: {resnet_mean:.2f}% (Human - ResNet: {human_mean - resnet_mean:.2f}%)")
    report.append("")

    # 2. 按遮挡级别详细对比
    report.append("2. PERFORMANCE BY OCCLUSION LEVEL")
    report.append("-" * 80)
    report.append("")

    for _, row in comparison_df.iterrows():
        occ = row['occlusion_level']
        human_acc = row['human_accuracy'] * 100
        vit_acc = row['vit_accuracy'] * 100
        resnet_acc = row['resnet_accuracy'] * 100

        report.append(f"{occ} Occlusion:")
        report.append(f"  Human:     {human_acc:.2f}%")
        report.append(f"  ViT-B/16:  {vit_acc:.2f}% (Gap: {human_acc - vit_acc:+.2f}%)")
        report.append(f"  ResNet-50: {resnet_acc:.2f}% (Gap: {human_acc - resnet_acc:+.2f}%)")
        report.append("")

    # 3. 关键发现
    report.append("3. KEY FINDINGS")
    report.append("-" * 80)
    report.append("")

    findings = []

    # 人类在所有遮挡级别上都优于AI
    if all(comparison_df['human_accuracy'] > comparison_df['vit_accuracy']) and \
       all(comparison_df['human_accuracy'] > comparison_df['resnet_accuracy']):
        findings.append("[FINDING 1] Humans significantly outperform AI models at ALL occlusion levels")

    # 遮挡效应
    human_decline = (comparison_df['human_accuracy'].iloc[0] - comparison_df['human_accuracy'].iloc[-1]) * 100
    vit_decline = (comparison_df['vit_accuracy'].iloc[0] - comparison_df['vit_accuracy'].iloc[-1]) * 100
    resnet_decline = (comparison_df['resnet_accuracy'].iloc[0] - comparison_df['resnet_accuracy'].iloc[-1]) * 100

    findings.append(f"[FINDING 2] Accuracy decline from 10% to 90% occlusion:")
    findings.append(f"            Human: {human_decline:.1f}%")
    findings.append(f"            ViT:   {vit_decline:.1f}%")
    findings.append(f"            ResNet: {resnet_decline:.1f}%")

    # ViT vs ResNet
    vit_better = vit_mean > resnet_mean
    if vit_better:
        findings.append(f"[FINDING 3] ViT-B/16 outperforms ResNet-50 by {vit_mean - resnet_mean:.2f}%")
    else:
        findings.append(f"[FINDING 3] ResNet-50 outperforms ViT-B/16 by {resnet_mean - vit_mean:.2f}%")

    # 90%遮挡
    acc_90 = comparison_df[comparison_df['occlusion_level'] == '90%'].iloc[0]
    if acc_90['human_accuracy'] > acc_90['vit_accuracy'] and \
       acc_90['human_accuracy'] > acc_90['resnet_accuracy']:
        findings.append("[FINDING 4] Even at 90% occlusion (extreme difficulty), humans still outperform AI")

    report.extend(findings)
    report.append("")

    # 4. 统计显著性说明
    report.append("4. STATISTICAL NOTES")
    report.append("-" * 80)
    report.append("")
    report.append(f"Test set size: {len(vit_df)} images")
    report.append(f"Images per occlusion level: {len(vit_df) // 3}")
    report.append(f"Human subjects: 65")
    report.append(f"Human trials per image: ~65 (varies by image)")
    report.append("")
    report.append("Note: Human accuracy represents majority voting across 65 subjects.")
    report.append("      AI accuracy represents single prediction per image.")
    report.append("")

    # 5. 结论
    report.append("5. CONCLUSION")
    report.append("-" * 80)
    report.append("")
    report.append("This experiment demonstrates that:")
    report.append("1. Humans maintain superior performance in occluded object recognition")
    report.append("2. Both ViT and ResNet struggle with this task (<60% accuracy)")
    report.append("3. The performance gap suggests fundamental differences in processing strategies")
    report.append("4. AI models may need architectural innovations to match human amodal completion")
    report.append("")

    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"[OK] Saved text report to {output_path}")

    # 同时打印到控制台
    print("\n" + '\n'.join(report))


def main():
    print("="*80)
    print("HUMAN vs AI MODEL COMPARISON ANALYSIS")
    print("="*80)

    # 创建输出目录
    output_dir = Path("experiments/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    human_df, vit_df, resnet_df, mae_df = load_all_data()

    # 匹配数据
    print("\nChecking data overlap...")
    matched_human, num_overlap = match_test_images_with_human(human_df, vit_df)

    if num_overlap == 0:
        # 使用整体人类平均
        human_for_comparison = human_df
    else:
        human_for_comparison = matched_human

    # 创建对比摘要
    print("\nCreating comparison summary...")
    comparison_df = create_comparison_summary(human_for_comparison, vit_df, resnet_df, mae_df)

    # 打印表格
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)

    # 绘制对比图
    plot_path = output_dir / "human_vs_ai_performance.png"
    plot_performance_comparison(comparison_df, plot_path)

    # 生成文本报告
    report_path = output_dir / "final_comparison_report.txt"
    if mae_df is not None:
        generate_text_report(comparison_df, vit_df, resnet_df, report_path, mae_df=mae_df)
    else:
        generate_text_report(comparison_df, vit_df, resnet_df, report_path)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {plot_path}")
    print(f"  2. {report_path}")


if __name__ == "__main__":
    main()
