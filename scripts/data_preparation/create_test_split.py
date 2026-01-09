"""
创建test数据集

从train_augmented中提取test split对应的图片（包括所有增强版本）
"""

import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_test_split():
    """从train_augmented提取test图片"""

    # 路径设置
    splits_dir = project_root / "data" / "splits"
    train_augmented_dir = project_root / "data" / "train_augmented"
    test_dir = project_root / "data" / "test"

    # 创建test目录
    test_dir.mkdir(parents=True, exist_ok=True)

    # 读取test split文件
    test_images_file = splits_dir / "test_images.txt"
    with open(test_images_file, 'r', encoding='utf-8') as f:
        test_images = [line.strip() for line in f if line.strip()]

    print(f"Test split包含 {len(test_images)} 张原始图片")

    # 统计信息
    total_copied = 0

    # 对每张test图片，复制所有增强版本
    for test_img in tqdm(test_images, desc="复制test图片"):
        # 解析文件名: Aircraft1_10%_16.jpg
        base_name = test_img.replace('.jpg', '')

        # 查找所有增强版本: Aircraft1_10%_16_aug*.jpg
        pattern = f"{base_name}_aug*.jpg"
        augmented_files = list(train_augmented_dir.glob(pattern))

        if not augmented_files:
            print(f"警告: 未找到 {test_img} 的增强版本")
            continue

        # 复制所有增强版本到test目录
        for aug_file in augmented_files:
            dest_file = test_dir / aug_file.name
            shutil.copy2(aug_file, dest_file)
            total_copied += 1

    print(f"\n完成!")
    print(f"原始test图片: {len(test_images)}")
    print(f"复制的增强图片: {total_copied}")
    print(f"每张图片平均增强数: {total_copied / len(test_images):.1f}")

    # 验证test目录
    test_files = list(test_dir.glob("*.jpg"))
    print(f"\nTest目录验证:")
    print(f"总文件数: {len(test_files)}")

    # 统计类别分布
    aircraft1 = len([f for f in test_files if 'Aircraft1' in f.name])
    aircraft2 = len([f for f in test_files if 'Aircraft2' in f.name])
    print(f"Aircraft1: {aircraft1}, Aircraft2: {aircraft2}")

    # 统计遮挡级别
    from collections import Counter
    occlusion = Counter([f.name.split('_')[1] for f in test_files])
    print(f"遮挡级别分布: {dict(occlusion)}")


if __name__ == '__main__':
    create_test_split()
