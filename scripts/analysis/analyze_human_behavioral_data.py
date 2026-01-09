"""
加载和分析OIID数据集的人类行为数据

功能：
1. 加载所有65个受试者的行为数据 (CSV文件)
2. 计算各遮挡级别的统计信息
3. 按图像统计人类表现
4. 按受试者统计
5. 保存汇总结果

输出：
- data/human_performance/human_all_trials.csv
- data/human_performance/human_by_occlusion.csv
- data/human_performance/human_by_image.csv
- data/human_performance/human_by_subject.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_all_subjects(behavioral_dir: str = "E:/Dataset/ds005226/derivatives/Behavioral_data") -> pd.DataFrame:
    """
    加载所有65个受试者的行为数据

    Args:
        behavioral_dir: 行为数据目录路径

    Returns:
        包含所有试次的DataFrame
    """
    behavioral_path = Path(behavioral_dir)

    if not behavioral_path.exists():
        raise FileNotFoundError(f"Behavioral data directory not found: {behavioral_dir}")

    print("Loading human behavioral data...")
    print(f"Source: {behavioral_dir}")

    all_data = []
    subject_files = sorted(behavioral_path.glob("sub-*.csv"))

    print(f"Found {len(subject_files)} subject files")

    for i, sub_file in enumerate(subject_files, 1):
        try:
            df = pd.read_csv(sub_file)
            # 添加受试者ID
            df['subject_id'] = sub_file.stem
            all_data.append(df)

            if i % 10 == 0:
                print(f"  Loaded {i}/{len(subject_files)} subjects...")
        except Exception as e:
            print(f"  Warning: Failed to load {sub_file.name}: {e}")

    if not all_data:
        raise ValueError("No data loaded! Check directory path.")

    all_trials = pd.concat(all_data, ignore_index=True)

    print(f"[OK] Loaded {len(all_trials)} trials from {len(subject_files)} subjects")
    print(f"  Columns: {list(all_trials.columns)}")

    return all_trials


def calculate_by_occlusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算各遮挡级别的统计数据

    Args:
        df: 包含所有试次的DataFrame

    Returns:
        按遮挡级别汇总的统计DataFrame
    """
    print("\nCalculating statistics by occlusion level...")

    # 确保occlusion_level是字符串类型
    df['occlusion_level'] = df['occlusion_level'].astype(str)

    stats = df.groupby('occlusion_level').agg({
        'Keypress_Score': ['mean', 'std', 'count'],
        'reaction_time (ms)': ['mean', 'std']
    }).round(4)

    stats.columns = ['accuracy', 'accuracy_std', 'num_trials', 'mean_rt', 'rt_std']

    # 重新排序（10%, 70%, 90%）
    occlusion_order = ['10%', '70%', '90%']
    stats = stats.reindex(occlusion_order)

    print("\nSummary by Occlusion Level:")
    print("="*80)
    print(stats.to_string())

    return stats.reset_index()


def calculate_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个受试者的统计数据

    Args:
        df: 包含所有试次的DataFrame

    Returns:
        按受试者汇总的统计DataFrame
    """
    print("\nCalculating statistics by subject...")

    subject_stats = []

    for subject_id in df['subject_id'].unique():
        subj_data = df[df['subject_id'] == subject_id]

        # 整体统计
        total_acc = subj_data['Keypress_Score'].mean()
        total_rt = subj_data['reaction_time (ms)'].mean()

        # 按遮挡级别统计
        occ_stats = {}
        for occ_level in ['10%', '70%', '90%']:
            occ_data = subj_data[subj_data['occlusion_level'] == occ_level]
            if len(occ_data) > 0:
                occ_stats[f'{occ_level}_acc'] = occ_data['Keypress_Score'].mean()
                occ_stats[f'{occ_level}_rt'] = occ_data['reaction_time (ms)'].mean()
            else:
                occ_stats[f'{occ_level}_acc'] = np.nan
                occ_stats[f'{occ_level}_rt'] = np.nan

        subject_stats.append({
            'subject_id': subject_id,
            'overall_accuracy': total_acc,
            'overall_mean_rt': total_rt,
            **occ_stats
        })

    stats_df = pd.DataFrame(subject_stats)

    print(f"[OK] Calculated statistics for {len(stats_df)} subjects")

    return stats_df


def calculate_by_image(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每张图像的人类表现

    Args:
        df: 包含所有试次的DataFrame

    Returns:
        按图像汇总的统计DataFrame
    """
    print("\nCalculating statistics by image...")

    image_stats = df.groupby('stimuli').agg({
        'Keypress_Score': ['mean', 'sum', 'count'],
        'reaction_time (ms)': ['mean', 'std']
    }).round(4)

    image_stats.columns = ['accuracy', 'correct_count', 'total_count', 'mean_rt', 'rt_std']

    # 提取遮挡级别和飞机类型
    image_df = image_stats.reset_index()

    # 从文件名提取信息
    def extract_info(filename):
        # Aircraft1_10%_2.jpg -> Aircraft1, 10%, 2
        parts = filename.replace('.jpg', '').split('_')
        aircraft = parts[0]
        occlusion = parts[1]
        image_num = parts[2] if len(parts) > 2 else ''
        return aircraft, occlusion, image_num

    image_df[['aircraft_type', 'occlusion_level', 'image_num']] = \
        pd.DataFrame(image_df['stimuli'].apply(extract_info).tolist(), index=image_df.index)

    print(f"[OK] Calculated statistics for {len(image_df)} unique images")

    return image_df


def quality_check(df: pd.DataFrame) -> Dict:
    """
    数据质量检查

    Args:
        df: 包含所有试次的DataFrame

    Returns:
        质量检查结果字典
    """
    print("\n" + "="*80)
    print("DATA QUALITY CHECK")
    print("="*80)

    results = {}

    # 1. 缺失值检查
    missing = df.isnull().sum()
    print("\n1. Missing Values:")
    print(missing)
    results['has_missing'] = missing.sum() > 0

    # 2. 试次数量检查
    print("\n2. Trial Counts:")
    print(f"   Total trials: {len(df)}")
    print(f"   Unique subjects: {df['subject_id'].nunique()}")
    print(f"   Unique images: {df['stimuli'].nunique()}")

    expected_trials_per_subject = 300  # 2 runs × 150 trials
    trial_counts = df.groupby('subject_id').size()
    low_count_subjects = trial_counts[trial_counts < expected_trials_per_subject * 0.9]
    print(f"   Subjects with <90% expected trials: {len(low_count_subjects)}")
    results['low_trial_subjects'] = low_count_subjects.tolist()

    # 3. 反应时异常值检查
    print("\n3. Reaction Time Outliers:")
    rt = df['reaction_time (ms)']
    too_fast = (rt < 200).sum()
    too_slow = (rt > 5000).sum()
    print(f"   RT < 200ms: {too_fast} ({too_fast/len(rt)*100:.2f}%)")
    print(f"   RT > 5000ms: {too_slow} ({too_slow/len(rt)*100:.2f}%)")
    results['rt_outliers'] = {'too_fast': int(too_fast), 'too_slow': int(too_slow)}

    # 4. 准确率检查
    print("\n4. Accuracy Check:")
    subject_acc = df.groupby('subject_id')['Keypress_Score'].mean()
    low_acc_subjects = subject_acc[subject_acc < 0.5]
    print(f"   Subjects with accuracy < 50%: {len(low_acc_subjects)}")
    if len(low_acc_subjects) > 0:
        print(f"   IDs: {low_acc_subjects.index.tolist()}")
    results['low_accuracy_subjects'] = low_acc_subjects.tolist()

    # 5. 遮挡级别分布
    print("\n5. Occlusion Level Distribution:")
    occ_dist = df['occlusion_level'].value_counts().sort_index()
    print(occ_dist)

    return results


def visualize_summary(df: pd.DataFrame, output_dir: Path):
    """
    生成可视化摘要图

    Args:
        df: 包含所有试次的DataFrame
        output_dir: 输出目录
    """
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 准确率vs遮挡级别
    occlusion_order = ['10%', '70%', '90%']
    acc_by_occ = df.groupby('occlusion_level')['Keypress_Score'].mean()[occlusion_order]

    axes[0, 0].bar(range(len(occlusion_order)), acc_by_occ.values * 100,
                   color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
    axes[0, 0].set_xticks(range(len(occlusion_order)))
    axes[0, 0].set_xticklabels([f'{occ}\nOcclusion' for occ in occlusion_order])
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Human Accuracy by Occlusion Level')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 100])

    # 2. 反应时vs遮挡级别
    rt_by_occ = df.groupby('occlusion_level')['reaction_time (ms)'].mean()[occlusion_order]

    axes[0, 1].bar(range(len(occlusion_order)), rt_by_occ.values,
                   color=['#3498db', '#9b59b6', '#34495e'], alpha=0.7)
    axes[0, 1].set_xticks(range(len(occlusion_order)))
    axes[0, 1].set_xticklabels([f'{occ}\nOcclusion' for occ in occlusion_order])
    axes[0, 1].set_ylabel('Reaction Time (ms)')
    axes[0, 1].set_title('Human Reaction Time by Occlusion Level')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. 受试者准确率分布
    subject_acc = df.groupby('subject_id')['Keypress_Score'].mean()
    axes[1, 0].hist(subject_acc.values * 100, bins=20, color='#1abc9c', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(subject_acc.mean() * 100, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {subject_acc.mean()*100:.1f}%')
    axes[1, 0].set_xlabel('Accuracy (%)')
    axes[1, 0].set_ylabel('Number of Subjects')
    axes[1, 0].set_title('Distribution of Subject Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. 准确率vs反应时散点图
    for occ in occlusion_order:
        occ_data = df[df['occlusion_level'] == occ]
        axes[1, 1].scatter(occ_data.groupby('subject_id')['Keypress_Score'].mean() * 100,
                          occ_data.groupby('subject_id')['reaction_time (ms)'].mean(),
                          label=f'{occ} occlusion', alpha=0.6, s=50)

    axes[1, 1].set_xlabel('Accuracy (%)')
    axes[1, 1].set_ylabel('Mean Reaction Time (ms)')
    axes[1, 1].set_title('Accuracy vs Reaction Time by Subject')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'human_behavior_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved visualization to {output_path}")

    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("OIID HUMAN BEHAVIORAL DATA ANALYSIS")
    print("="*80)

    # 创建输出目录
    output_dir = Path("data/human_performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    all_trials = load_all_subjects()

    # 2. 数据质量检查
    quality_results = quality_check(all_trials)

    # 3. 计算统计
    by_occlusion = calculate_by_occlusion(all_trials)
    by_subject = calculate_by_subject(all_trials)
    by_image = calculate_by_image(all_trials)

    # 4. 保存结果
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    all_trials.to_csv(output_dir / "human_all_trials.csv", index=False)
    print(f"[OK] Saved: {output_dir / 'human_all_trials.csv'} ({len(all_trials)} rows)")

    by_occlusion.to_csv(output_dir / "human_by_occlusion.csv", index=False)
    print(f"[OK] Saved: {output_dir / 'human_by_occlusion.csv'} ({len(by_occlusion)} rows)")

    by_subject.to_csv(output_dir / "human_by_subject.csv", index=False)
    print(f"[OK] Saved: {output_dir / 'human_by_subject.csv'} ({len(by_subject)} rows)")

    by_image.to_csv(output_dir / "human_by_image.csv", index=False)
    print(f"[OK] Saved: {output_dir / 'human_by_image.csv'} ({len(by_image)} rows)")

    # 5. 可视化
    visualize_summary(all_trials, output_dir)

    # 6. 最终摘要
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nSummary Statistics:")
    print(f"  Total trials: {len(all_trials)}")
    print(f"  Total subjects: {all_trials['subject_id'].nunique()}")
    print(f"  Total images: {all_trials['stimuli'].nunique()}")
    print(f"\nAccuracy by occlusion level:")
    for _, row in by_occlusion.iterrows():
        print(f"  {row['occlusion_level']}: {row['accuracy']*100:.2f}% "
              f"(±{row['accuracy_std']*100:.2f}%, n={int(row['num_trials'])})")

    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()
