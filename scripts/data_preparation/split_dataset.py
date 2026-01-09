"""
数据集划分脚本

功能:
1. 将原始图片复制到验证集
2. 将增强图片复制到训练集
3. 验证数据集平衡性

用法:
python split_dataset.py \
    --original-dir "e:/Dataset/ds005226/derivatives/stimuli_dataset/stimuli_original" \
    --augmented-dir "data/train_augmented" \
    --train-dir "data/train" \
    --val-dir "data/val"
"""

import os
import sys
from pathlib import Path
import argparse
import shutil
from collections import Counter
from tqdm import tqdm
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def split_dataset(original_dir, augmented_dir, train_dir, val_dir):
    """
    划分数据集

    Args:
        original_dir: 原始图片目录
        augmented_dir: 增强图片目录
        train_dir: 训练集输出目录
        val_dir: 验证集输出目录
    """
    original_path = Path(original_dir)
    augmented_path = Path(augmented_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)

    # 创建输出目录
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("数据集划分")
    print("="*60)

    # Step 1: 复制原始图片到验证集
    print("\n[1/2] 复制原始图片到验证集...")

    original_files = list(original_path.glob('*.jpg'))
    print(f"找到 {len(original_files)} 张原始图片")

    val_stats = Counter()

    for img_file in tqdm(original_files, desc="复制到验证集"):
        # 解析文件名: Aircraft1_10%_1_original.jpg
        filename = img_file.stem
        parts = filename.split('_')

        if len(parts) < 3:
            print(f"警告: 跳过文件 {img_file.name} (文件名格式不正确)")
            continue

        aircraft_class = parts[0]  # Aircraft1 或 Aircraft2
        occlusion_level = parts[1]  # 10%, 70%, 90%

        # 生成新文件名 (去掉_original后缀)
        # Aircraft1_10%_1_original.jpg -> Aircraft1_10%_1.jpg
        new_filename = f"{aircraft_class}_{occlusion_level}_{parts[2]}.jpg"
        output_file = val_path / new_filename

        # 复制文件
        shutil.copy2(img_file, output_file)

        # 统计
        val_stats[f"{aircraft_class}_{occlusion_level}"] += 1

    print(f"验证集: {len(original_files)} 张图片")

    # Step 2: 复制增强图片到训练集
    print("\n[2/2] 复制增强图片到训练集...")

    augmented_files = list(augmented_path.glob('*.jpg'))
    print(f"找到 {len(augmented_files)} 张增强图片")

    train_stats = Counter()

    for img_file in tqdm(augmented_files, desc="复制到训练集"):
        # 解析文件名: Aircraft1_10%_1_aug0.jpg
        filename = img_file.stem
        parts = filename.split('_')

        if len(parts) < 4:
            print(f"警告: 跳过文件 {img_file.name} (文件名格式不正确)")
            continue

        aircraft_class = parts[0]  # Aircraft1 或 Aircraft2
        occlusion_level = parts[1]  # 10%, 70%, 90%

        # 直接复制 (保持原文件名)
        output_file = train_path / img_file.name
        shutil.copy2(img_file, output_file)

        # 统计
        train_stats[f"{aircraft_class}_{occlusion_level}"] += 1

    print(f"训练集: {len(augmented_files)} 张图片")

    # Step 3: 生成统计报告
    print("\n" + "="*60)
    print("数据集统计")
    print("="*60)

    # 训练集统计
    print("\n训练集:")
    print(f"  总计: {sum(train_stats.values())} 张")
    print(f"\n  按类别和遮挡等级:")
    for key in sorted(train_stats.keys()):
        print(f"    {key}: {train_stats[key]} 张")

    # 验证集统计
    print("\n验证集:")
    print(f"  总计: {sum(val_stats.values())} 张")
    print(f"\n  按类别和遮挡等级:")
    for key in sorted(val_stats.keys()):
        print(f"    {key}: {val_stats[key]} 张")

    # 检查平衡性
    print("\n" + "="*60)
    print("平衡性检查")
    print("="*60)

    # 训练集平衡性
    print("\n训练集:")
    aircraft1_train = sum(v for k, v in train_stats.items() if k.startswith('Aircraft1'))
    aircraft2_train = sum(v for k, v in train_stats.items() if k.startswith('Aircraft2'))
    print(f"  Aircraft1: {aircraft1_train} 张 ({aircraft1_train/sum(train_stats.values())*100:.1f}%)")
    print(f"  Aircraft2: {aircraft2_train} 张 ({aircraft2_train/sum(train_stats.values())*100:.1f}%)")

    occlusion_10_train = sum(v for k, v in train_stats.items() if '10%' in k)
    occlusion_70_train = sum(v for k, v in train_stats.items() if '70%' in k)
    occlusion_90_train = sum(v for k, v in train_stats.items() if '90%' in k)
    print(f"\n  10%遮挡: {occlusion_10_train} 张 ({occlusion_10_train/sum(train_stats.values())*100:.1f}%)")
    print(f"  70%遮挡: {occlusion_70_train} 张 ({occlusion_70_train/sum(train_stats.values())*100:.1f}%)")
    print(f"  90%遮挡: {occlusion_90_train} 张 ({occlusion_90_train/sum(train_stats.values())*100:.1f}%)")

    # 验证集平衡性
    print("\n验证集:")
    aircraft1_val = sum(v for k, v in val_stats.items() if k.startswith('Aircraft1'))
    aircraft2_val = sum(v for k, v in val_stats.items() if k.startswith('Aircraft2'))
    print(f"  Aircraft1: {aircraft1_val} 张 ({aircraft1_val/sum(val_stats.values())*100:.1f}%)")
    print(f"  Aircraft2: {aircraft2_val} 张 ({aircraft2_val/sum(val_stats.values())*100:.1f}%)")

    occlusion_10_val = sum(v for k, v in val_stats.items() if '10%' in k)
    occlusion_70_val = sum(v for k, v in val_stats.items() if '70%' in k)
    occlusion_90_val = sum(v for k, v in val_stats.items() if '90%' in k)
    print(f"\n  10%遮挡: {occlusion_10_val} 张 ({occlusion_10_val/sum(val_stats.values())*100:.1f}%)")
    print(f"  70%遮挡: {occlusion_70_val} 张 ({occlusion_70_val/sum(val_stats.values())*100:.1f}%)")
    print(f"  90%遮挡: {occlusion_90_val} 张 ({occlusion_90_val/sum(val_stats.values())*100:.1f}%)")

    # 保存统计信息
    stats = {
        'train': {
            'total': sum(train_stats.values()),
            'by_class': {
                'Aircraft1': aircraft1_train,
                'Aircraft2': aircraft2_train
            },
            'by_occlusion': {
                '10%': occlusion_10_train,
                '70%': occlusion_70_train,
                '90%': occlusion_90_train
            },
            'detailed': dict(train_stats)
        },
        'val': {
            'total': sum(val_stats.values()),
            'by_class': {
                'Aircraft1': aircraft1_val,
                'Aircraft2': aircraft2_val
            },
            'by_occlusion': {
                '10%': occlusion_10_val,
                '70%': occlusion_70_val,
                '90%': occlusion_90_val
            },
            'detailed': dict(val_stats)
        }
    }

    stats_file = Path('data') / 'dataset_split_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n统计信息已保存到: {stats_file}")
    print("\n✅ 数据集划分完成!")


def main():
    parser = argparse.ArgumentParser(description='数据集划分脚本')
    parser.add_argument('--original-dir', type=str, required=True,
                        help='原始图片目录')
    parser.add_argument('--augmented-dir', type=str, required=True,
                        help='增强图片目录')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='训练集输出目录')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='验证集输出目录')

    args = parser.parse_args()

    split_dataset(
        original_dir=args.original_dir,
        augmented_dir=args.augmented_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir
    )


if __name__ == '__main__':
    main()
