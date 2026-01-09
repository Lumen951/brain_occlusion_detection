"""
遮挡生成脚本 (Python版的addmask.m)

功能:
1. 读取增强后的图片
2. 检测飞机区域 (非黑色背景)
3. 在飞机区域随机添加10×10像素的黑色方块
4. 根据文件名中的遮挡等级 (10%, 70%, 90%) 控制覆盖率
5. 保存带遮挡的图片

用法:
python add_occlusion.py \
    --input-dir "data/augmented_images" \
    --output-dir "data/train_augmented" \
    --mask-size 10 \
    --occlusion-levels 0.1 0.7 0.9
"""

import os
import sys
from pathlib import Path
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class OcclusionGenerator:
    """遮挡生成器"""

    def __init__(self, mask_size=10, seed=42):
        """
        初始化

        Args:
            mask_size: 遮挡方块大小 (像素)
            seed: 随机种子
        """
        self.mask_size = mask_size
        self.seed = seed

    def detect_plane_region(self, image):
        """
        检测飞机区域 (非黑色背景)

        Args:
            image: PIL Image对象

        Returns:
            plane_mask: numpy数组, True表示飞机区域, False表示背景
        """
        # 转换为numpy数组
        img_array = np.array(image)

        # 转换为灰度图
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # 非黑色区域即为飞机
        plane_mask = gray > 0

        return plane_mask

    def add_random_occlusion(self, image, coverage=0.1, seed=None):
        """
        在飞机区域添加随机遮挡

        Args:
            image: PIL Image对象
            coverage: 覆盖率 (0.1 = 10%, 0.7 = 70%, 0.9 = 90%)
            seed: 随机种子

        Returns:
            masked_image: 带遮挡的PIL Image对象
            mask: 遮挡mask (numpy数组)
        """
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 转换为numpy数组
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # 检测飞机区域
        plane_mask = self.detect_plane_region(image)

        # 计算飞机区域的总像素数
        plane_pixels = np.sum(plane_mask)

        if plane_pixels == 0:
            print("警告: 未检测到飞机区域")
            return image, np.zeros((height, width), dtype=bool)

        # 计算需要覆盖的像素数
        target_pixels = int(coverage * plane_pixels)

        # 创建遮挡mask (初始为全False)
        occlusion_mask = np.zeros((height, width), dtype=bool)

        # 获取飞机像素的坐标
        plane_coords = np.argwhere(plane_mask)  # [[y1, x1], [y2, x2], ...]

        # 随机添加遮挡方块
        covered_pixels = 0
        attempts = 0
        max_attempts = 10000

        while covered_pixels < target_pixels and attempts < max_attempts:
            # 随机选择一个中心点 (在飞机区域内)
            idx = random.randint(0, len(plane_coords) - 1)
            center_y, center_x = plane_coords[idx]

            # 计算方块边界
            x1 = max(0, center_x - self.mask_size // 2)
            x2 = min(width, center_x + self.mask_size // 2)
            y1 = max(0, center_y - self.mask_size // 2)
            y2 = min(height, center_y + self.mask_size // 2)

            # 计算这个区域内的飞机像素
            block_mask = np.zeros((height, width), dtype=bool)
            block_mask[y1:y2, x1:x2] = True
            valid_pixels = block_mask & plane_mask

            # 计算新增的覆盖像素 (之前未被覆盖的)
            new_pixels = valid_pixels & ~occlusion_mask
            num_new_pixels = np.sum(new_pixels)

            # 如果这个方块没有覆盖任何新的飞机像素,跳过
            if num_new_pixels == 0:
                attempts += 1
                continue

            # 添加遮挡 (只标记新增的覆盖区域)
            occlusion_mask[new_pixels] = True

            # 更新覆盖像素计数
            covered_pixels = np.sum(occlusion_mask)

            attempts = 0  # 重置尝试计数器

        # 应用遮挡到图像 (将遮挡区域设为黑色)
        masked_array = img_array.copy()
        masked_array[occlusion_mask] = [0, 0, 0]

        # 转换回PIL Image
        masked_image = Image.fromarray(masked_array)

        # 计算实际覆盖率
        actual_coverage = covered_pixels / plane_pixels if plane_pixels > 0 else 0

        return masked_image, occlusion_mask, actual_coverage

    def process_dataset(self, input_dir, output_dir):
        """
        处理整个数据集

        Args:
            input_dir: 输入目录 (增强后的图片)
            output_dir: 输出目录 (带遮挡的图片)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有图片文件
        image_files = list(input_path.glob('*.jpg'))
        print(f"找到 {len(image_files)} 张增强图片")

        # 统计信息
        stats = {
            'total_processed': 0,
            'coverage_stats': {
                '10%': {'target': 0.1, 'actual': []},
                '70%': {'target': 0.7, 'actual': []},
                '90%': {'target': 0.9, 'actual': []}
            }
        }

        # 处理每张图片
        for img_file in tqdm(image_files, desc="添加遮挡"):
            # 读取图片
            image = Image.open(img_file)

            # 解析文件名: Aircraft1_10%_1_aug0.jpg
            filename = img_file.stem
            parts = filename.split('_')

            if len(parts) < 4:
                print(f"警告: 跳过文件 {img_file.name} (文件名格式不正确)")
                continue

            aircraft_class = parts[0]  # Aircraft1 或 Aircraft2
            occlusion_level = parts[1]  # 10%, 70%, 90%
            image_id = parts[2]  # 1, 2, 3, ...
            aug_id = parts[3]  # aug0, aug1, ...

            # 确定覆盖率
            if occlusion_level == '10%':
                coverage = 0.1
            elif occlusion_level == '70%':
                coverage = 0.7
            elif occlusion_level == '90%':
                coverage = 0.9
            else:
                print(f"警告: 未知的遮挡等级 {occlusion_level}, 跳过")
                continue

            # 生成随机种子 (基于文件名,确保可复现)
            seed = self.seed + hash(filename) % 100000

            # 添加遮挡
            masked_image, mask, actual_coverage = self.add_random_occlusion(
                image,
                coverage=coverage,
                seed=seed
            )

            # 生成输出文件名
            # 格式: Aircraft1_10%_1_aug0.jpg (保持原文件名)
            output_file = output_path / img_file.name

            # 保存
            masked_image.save(output_file, quality=95)

            # 记录统计信息
            stats['coverage_stats'][occlusion_level]['actual'].append(actual_coverage)
            stats['total_processed'] += 1

        # 计算平均覆盖率
        for level in ['10%', '70%', '90%']:
            actual_list = stats['coverage_stats'][level]['actual']
            if actual_list:
                stats['coverage_stats'][level]['mean'] = np.mean(actual_list)
                stats['coverage_stats'][level]['std'] = np.std(actual_list)
                stats['coverage_stats'][level]['min'] = np.min(actual_list)
                stats['coverage_stats'][level]['max'] = np.max(actual_list)

        # 保存统计信息
        stats_file = output_path / 'occlusion_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\n遮挡生成完成!")
        print(f"处理图片: {stats['total_processed']}")
        print(f"\n覆盖率统计:")
        for level in ['10%', '70%', '90%']:
            if 'mean' in stats['coverage_stats'][level]:
                mean = stats['coverage_stats'][level]['mean']
                std = stats['coverage_stats'][level]['std']
                print(f"  {level}: {mean:.3f} ± {std:.3f}")
        print(f"\n统计信息已保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='遮挡生成脚本')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入目录 (增强后的图片)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录 (带遮挡的图片)')
    parser.add_argument('--mask-size', type=int, default=10,
                        help='遮挡方块大小 (默认10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认42)')

    args = parser.parse_args()

    # 创建遮挡生成器
    generator = OcclusionGenerator(mask_size=args.mask_size, seed=args.seed)

    # 处理数据集
    generator.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
