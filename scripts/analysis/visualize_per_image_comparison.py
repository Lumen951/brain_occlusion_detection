"""
按图像对比人类、ViT和ResNet的识别正确率

生成可视化展示每张测试图像在三方（人类、ViT、ResNet）的识别结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_data():
    """加载所有数据"""
    print("Loading data...")

    # 人类数据
    human_df = pd.read_csv("data/human_performance/human_by_image.csv")
    print(f"  Human data: {len(human_df)} images")

    # ViT预测
    vit_df = pd.read_csv("experiments/analysis/vit_b16_test_predictions.csv")
    print(f"  ViT predictions: {len(vit_df)} images")

    # ResNet预测
    resnet_df = pd.read_csv("experiments/analysis/resnet50_test_predictions.csv")
    print(f"  ResNet predictions: {len(resnet_df)} images")

    return human_df, vit_df, resnet_df


def match_images(human_df, vit_df, resnet_df):
    """匹配所有数据"""
    print("\nMatching images...")

    # 获取测试集图像
    test_images = set(vit_df['image_name'])
    human_images = set(human_df['stimuli'])

    overlap = test_images & human_images
    print(f"  Overlapping images: {len(overlap)}")

    # 过滤出重叠的图像
    human_matched = human_df[human_df['stimuli'].isin(overlap)].copy()
    vit_matched = vit_df[vit_df['image_name'].isin(overlap)].copy()
    resnet_matched = resnet_df[resnet_df['image_name'].isin(overlap)].copy()

    # 合并数据
    merged = human_matched[['stimuli', 'accuracy', 'occlusion_level']].copy()
    merged = merged.merge(
        vit_matched[['image_name', 'correct']],
        left_on='stimuli',
        right_on='image_name',
        how='left'
    )
    merged = merged.rename(columns={'correct': 'vit_correct'})

    merged = merged.merge(
        resnet_matched[['image_name', 'correct']],
        left_on='stimuli',
        right_on='image_name',
        how='left'
    )
    merged = merged.rename(columns={'correct': 'resnet_correct'})

    # 将人类准确率转换为正确率（>0.5视为正确）
    merged['human_correct'] = (merged['accuracy'] > 0.5).astype(int)

    # 将 AI 模型的布尔值转换为 int 类型
    merged['vit_correct'] = merged['vit_correct'].astype(int)
    merged['resnet_correct'] = merged['resnet_correct'].astype(int)

    # 排序：按遮挡级别，然后按图像名称
    merged = merged.sort_values(['occlusion_level', 'stimuli'])

    return merged




def plot_heatmap(merged_df, output_path):
    """绘制热力图"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # 准备数据矩阵
    df = merged_df.copy()
    df = df.sort_values(['occlusion_level', 'stimuli'])

    # 创建矩阵：行=图像，列=模型
    matrix_data = df[['human_correct', 'vit_correct', 'resnet_correct']].T
    matrix_data.columns = [f"Img_{i+1}" for i in range(len(df))]

    # 按遮挡级别分组标注
    occ_counts = df['occlusion_level'].value_counts().sort_index()
    x_labels = []
    for occ, count in occ_counts.items():
        x_labels.extend([f"{occ}\n{i+1}" for i in range(count)])

    # 绘制热力图
    sns.heatmap(matrix_data,
                annot=True,
                fmt='d',
                cmap='RdYlGn',
                cbar_kws={'label': 'Correct (1=Yes, 0=No)'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax,
                xticklabels=x_labels,
                yticklabels=['Human', 'ViT-B/16', 'ResNet-50'])

    ax.set_xlabel('Test Images (grouped by occlusion level)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Per-Image Recognition Accuracy: Human vs AI Models',
                fontsize=14, fontweight='bold')

    plt.xticks(rotation=0, fontsize=7)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved heatmap to {output_path}")
    plt.close()


def plot_agreement_analysis(merged_df, output_path):
    """绘制一致性分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df = merged_df.copy()

    # 1. 韦恩图风格的一致性统计
    ax1 = axes[0, 0]

    # 计算各种情况的数量
    all_correct = ((df['human_correct'] == 1) &
                   (df['vit_correct'] == 1) &
                   (df['resnet_correct'] == 1)).sum()

    human_vit_correct = ((df['human_correct'] == 1) &
                         (df['vit_correct'] == 1) &
                         (df['resnet_correct'] == 0)).sum()

    human_resnet_correct = ((df['human_correct'] == 1) &
                           (df['resnet_correct'] == 1) &
                           (df['vit_correct'] == 0)).sum()

    ai_only_correct = ((df['human_correct'] == 0) &
                      (df['vit_correct'] == 1) &
                      (df['resnet_correct'] == 1)).sum()

    all_wrong = ((df['human_correct'] == 0) &
                (df['vit_correct'] == 0) &
                (df['resnet_correct'] == 0)).sum()

    categories = ['All\nCorrect', 'Human+ViT\nOnly', 'Human+ResNet\nOnly',
                  'AI Only\nCorrect', 'All\nWrong']
    counts = [all_correct, human_vit_correct, human_resnet_correct,
              ai_only_correct, all_wrong]
    colors_cat = ['#27ae60', '#f39c12', '#3498db', '#9b59b6', '#e74c3c']

    bars = ax1.bar(categories, counts, color=colors_cat, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    ax1.set_title('Agreement Analysis: How Often Do They Agree?',
                fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)

    # 2. 按遮挡级别的一致性
    ax2 = axes[0, 1]

    occlusion_agreement = []
    for occ in ['10%', '70%', '90%']:
        occ_df = df[df['occlusion_level'] == occ]

        # 计算完全一致的比例
        all_agree = ((occ_df['human_correct'] == occ_df['vit_correct']) &
                     (occ_df['vit_correct'] == occ_df['resnet_correct'])).sum()

        total = len(occ_df)
        occlusion_agreement.append({
            'occlusion': occ,
            'all_agree': all_agree,
            'all_agree_pct': all_agree / total * 100 if total > 0 else 0,
            'total': total
        })

    x = range(len(occlusion_agreement))
    pct = [item['all_agree_pct'] for item in occlusion_agreement]

    bars = ax2.bar(x, pct, color='#8e44ad', alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels([item['occlusion'] for item in occlusion_agreement])
    ax2.set_ylabel('Full Agreement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Agreement Rate by Occlusion Level',
                fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. 人类正确但AI错误
    ax3 = axes[1, 0]

    human_wrong_vit = ((df['human_correct'] == 1) & (df['vit_correct'] == 0)).sum()
    human_wrong_resnet = ((df['human_correct'] == 1) & (df['resnet_correct'] == 0)).sum()

    categories = ['Human > ViT\n(Human Right, ViT Wrong)',
                  'Human > ResNet\n(Human Right, ResNet Wrong)']
    counts = [human_wrong_vit, human_wrong_resnet]
    colors_cat = ['#e67e22', '#2980b9']

    bars = ax3.bar(categories, counts, color=colors_cat, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    ax3.set_title('Where Humans Outperform AI',
                fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)

    # 4. AI正确但人类错误
    ax4 = axes[1, 1]

    vit_wrong_human = ((df['vit_correct'] == 1) & (df['human_correct'] == 0)).sum()
    resnet_wrong_human = ((df['resnet_correct'] == 1) & (df['human_correct'] == 0)).sum()

    categories = ['ViT > Human\n(ViT Right, Human Wrong)',
                  'ResNet > Human\n(ResNet Right, Human Wrong)']
    counts = [vit_wrong_human, resnet_wrong_human]
    colors_cat = ['#c0392b', '#2980b9']

    bars = ax4.bar(categories, counts, color=colors_cat, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    ax4.set_title('Where AI Outperforms Humans',
                fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved agreement analysis to {output_path}")
    plt.close()


def generate_per_image_summary_table(merged_df, output_path):
    """生成每张图像的汇总表格"""
    df = merged_df.copy()

    # 创建汇总表
    summary = df[['stimuli', 'occlusion_level', 'human_correct',
                   'vit_correct', 'resnet_correct']].copy()

    # 添加汇总列
    summary['all_correct'] = ((summary['human_correct'] == 1) &
                              (summary['vit_correct'] == 1) &
                              (summary['resnet_correct'] == 1)).astype(int)

    summary['all_wrong'] = ((summary['human_correct'] == 0) &
                            (summary['vit_correct'] == 0) &
                            (summary['resnet_correct'] == 0)).astype(int)

    # 重命名列
    summary = summary.rename(columns={
        'stimuli': 'Image',
        'occlusion_level': 'Occlusion',
        'human_correct': 'Human',
        'vit_correct': 'ViT',
        'resnet_correct': 'ResNet'
    })

    # 重新排序列
    summary = summary[['Image', 'Occlusion', 'Human', 'ViT', 'ResNet',
                       'all_correct', 'all_wrong']]

    # 保存
    summary.to_csv(output_path, index=False)
    print(f"[OK] Saved summary table to {output_path}")

    # 打印统计
    print("\n" + "="*80)
    print("PER-IMAGE COMPARISON SUMMARY")
    print("="*80)
    print(f"Total images compared: {len(summary)}")
    print(f"\nAll three correct: {summary['all_correct'].sum()} images")
    print(f"All three wrong: {summary['all_wrong'].sum()} images")
    print(f"\nCorrect by model:")
    print(f"  Human:  {summary['Human'].sum()} / {len(summary)} ({summary['Human'].mean()*100:.1f}%)")
    print(f"  ViT:     {summary['ViT'].sum()} / {len(summary)} ({summary['ViT'].mean()*100:.1f}%)")
    print(f"  ResNet:  {summary['ResNet'].sum()} / {len(summary)} ({summary['ResNet'].mean()*100:.1f}%)")


def main():
    print("="*80)
    print("PER-IMAGE RECOGNITION COMPARISON")
    print("Human vs ViT-B/16 vs ResNet-50")
    print("="*80)

    # 创建输出目录
    output_dir = Path("experiments/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    human_df, vit_df, resnet_df = load_data()

    # 匹配图像
    merged_df = match_images(human_df, vit_df, resnet_df)

    # 生成可视化
    print("\nGenerating visualizations...")

    # 1. 热力图
    plot_heatmap(
        merged_df,
        output_dir / "per_image_comparison_heatmap.png"
    )

    # 2. 一致性分析
    plot_agreement_analysis(
        merged_df,
        output_dir / "per_image_agreement_analysis.png"
    )

    # 3. 汇总表格
    generate_per_image_summary_table(
        merged_df,
        output_dir / "per_image_comparison_summary.csv"
    )

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. experiments/analysis/per_image_comparison_heatmap.png")
    print(f"  2. experiments/analysis/per_image_agreement_analysis.png")
    print(f"  3. experiments/analysis/per_image_comparison_summary.csv")
    print("\nYou can view the detailed comparison in the CSV file or visualizations.")


if __name__ == "__main__":
    main()
