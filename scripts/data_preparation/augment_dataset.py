"""
数据增强脚本

功能:
1. 读取原始图片
2. 应用几何变换 (旋转, 缩放, 平移, 翻转)
3. 应用颜色变换 (亮度, 对比度, 饱和度, 色调)
4. 生成增强图片

用法:
python augment_dataset.py \
    --input-dir "e:/Dataset/ds005226/derivatives/stimuli_dataset/stimuli_original" \
    --output-dir "data/augmented_images" \
    --num-augmentations 33 \
    --seed 42
"""

import os
import sys
from pathlib import Path
import argparse
import random
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DataAugmentor:
    """数据增强器"""

    def __init__(self, seed=42):
        """
        初始化

        Args:
            seed: 随机种子
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # 几何变换参数 (所有图片都使用)
        self.rotation_angles = [-30, -20, -10, -5, 0, 5, 10, 20, 30]
        self.scale_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.translation_range = [-20, -10, 0, 10, 20]

        # 颜色变换参数 (缩小范围,更保守)
        self.brightness_factors = [0.85, 0.95, 1.0, 1.05, 1.15]  # 缩小范围
        self.contrast_factors = [0.85, 0.95, 1.0, 1.05, 1.15]    # 缩小范围
        self.saturation_factors = [0.85, 0.95, 1.0, 1.05, 1.15]  # 缩小范围
        self.hue_shifts = [0]  # 禁用色调偏移 (最危险)

        # 颜色增强概率 (30%的图片使用颜色变换)
        self.color_augmentation_prob = 0.3

    def apply_geometric_transforms(self, image, rotation=0, scale=1.0,
                                    translate_x=0, translate_y=0, flip=False):
        """
        应用几何变换

        Args:
            image: PIL Image对象
            rotation: 旋转角度 (度)
            scale: 缩放比例
            translate_x: X方向平移 (像素)
            translate_y: Y方向平移 (像素)
            flip: 是否水平翻转

        Returns:
            变换后的PIL Image对象
        """
        # 1. 旋转
        if rotation != 0:
            image = image.rotate(rotation, resample=Image.BICUBIC, expand=False)

        # 2. 缩放
        if scale != 1.0:
            width, height = image.size
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.BICUBIC)

            # 如果缩放后尺寸变大,裁剪到原始尺寸
            if scale > 1.0:
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                image = image.crop((left, top, left + width, top + height))
            # 如果缩放后尺寸变小,填充到原始尺寸
            elif scale < 1.0:
                new_image = Image.new('RGB', (width, height), (0, 0, 0))
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                new_image.paste(image, (left, top))
                image = new_image

        # 3. 平移
        if translate_x != 0 or translate_y != 0:
            width, height = image.size
            new_image = Image.new('RGB', (width, height), (0, 0, 0))

            # 计算粘贴位置
            paste_x = max(0, translate_x)
            paste_y = max(0, translate_y)

            # 计算裁剪区域
            crop_x = max(0, -translate_x)
            crop_y = max(0, -translate_y)
            crop_width = min(width, width - abs(translate_x))
            crop_height = min(height, height - abs(translate_y))

            cropped = image.crop((crop_x, crop_y,
                                  crop_x + crop_width,
                                  crop_y + crop_height))
            new_image.paste(cropped, (paste_x, paste_y))
            image = new_image

        # 4. 水平翻转
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def apply_color_transforms(self, image, brightness=1.0, contrast=1.0,
                                saturation=1.0, hue=0.0):
        """
        应用颜色变换

        Args:
            image: PIL Image对象
            brightness: 亮度因子
            contrast: 对比度因子
            saturation: 饱和度因子
            hue: 色调偏移

        Returns:
            变换后的PIL Image对象
        """
        # 1. 亮度
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)

        # 2. 对比度
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        # 3. 饱和度
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)

        # 4. 色调 (需要转换到HSV空间)
        if hue != 0.0:
            # 转换到numpy数组
            img_array = np.array(image).astype(np.float32) / 255.0

            # 转换到HSV
            from colorsys import rgb_to_hsv, hsv_to_rgb

            # 对每个像素应用色调偏移
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    r, g, b = img_array[i, j]
                    h, s, v = rgb_to_hsv(r, g, b)
                    h = (h + hue) % 1.0  # 色调循环
                    r, g, b = hsv_to_rgb(h, s, v)
                    img_array[i, j] = [r, g, b]

            # 转换回PIL Image
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array)

        return image

    def augment_image(self, image, aug_id):
        """
        对单张图片应用随机增强

        Args:
            image: PIL Image对象
            aug_id: 增强ID (用于随机种子)

        Returns:
            增强后的PIL Image对象, 增强参数字典
        """
        # 设置随机种子 (确保可复现)
        random.seed(self.seed + aug_id)
        np.random.seed(self.seed + aug_id)

        # 随机选择几何变换参数
        rotation = random.choice(self.rotation_angles)
        scale = random.choice(self.scale_factors)
        translate_x = random.choice(self.translation_range)
        translate_y = random.choice(self.translation_range)
        flip = random.choice([True, False])

        # 应用几何变换 (所有图片都使用)
        augmented = self.apply_geometric_transforms(
            image,
            rotation=rotation,
            scale=scale,
            translate_x=translate_x,
            translate_y=translate_y,
            flip=flip
        )

        # 随机决定是否应用颜色变换 (30%概率)
        use_color_aug = random.random() < self.color_augmentation_prob

        if use_color_aug:
            # 应用颜色变换
            brightness = random.choice(self.brightness_factors)
            contrast = random.choice(self.contrast_factors)
            saturation = random.choice(self.saturation_factors)
            hue = random.choice(self.hue_shifts)

            augmented = self.apply_color_transforms(
                augmented,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
        else:
            # 不使用颜色变换,保持原始颜色
            brightness = 1.0
            contrast = 1.0
            saturation = 1.0
            hue = 0.0

        # 记录增强参数
        params = {
            'rotation': rotation,
            'scale': scale,
            'translate_x': translate_x,
            'translate_y': translate_y,
            'flip': flip,
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'hue': hue,
            'use_color_aug': use_color_aug  # 记录是否使用了颜色增强
        }

        return augmented, params

    def process_dataset(self, input_dir, output_dir, num_augmentations=33):
        """
        处理整个数据集

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            num_augmentations: 每张图片生成的增强数量
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有图片文件
        image_files = list(input_path.glob('*.jpg'))
        print(f"找到 {len(image_files)} 张原始图片")

        # 统计信息
        stats = {
            'total_original': len(image_files),
            'total_augmented': 0,
            'color_augmented': 0,  # 使用颜色增强的数量
            'geometric_only': 0,   # 仅几何增强的数量
            'augmentation_params': []
        }

        # 处理每张图片
        for img_file in tqdm(image_files, desc="增强图片"):
            # 读取图片
            image = Image.open(img_file)

            # 解析文件名: Aircraft1_10%_1_original.jpg
            filename = img_file.stem  # 去掉扩展名
            parts = filename.split('_')

            if len(parts) < 3:
                print(f"警告: 跳过文件 {img_file.name} (文件名格式不正确)")
                continue

            aircraft_class = parts[0]  # Aircraft1 或 Aircraft2
            occlusion_level = parts[1]  # 10%, 70%, 90%
            image_id = parts[2]  # 1, 2, 3, ...

            # 生成多个增强版本
            for aug_id in range(num_augmentations):
                # 应用增强
                augmented, params = self.augment_image(image, aug_id)

                # 生成输出文件名
                # 格式: Aircraft1_10%_1_aug0.jpg
                output_filename = f"{aircraft_class}_{occlusion_level}_{image_id}_aug{aug_id}.jpg"
                output_file = output_path / output_filename

                # 保存
                augmented.save(output_file, quality=95)

                # 记录参数
                stats['augmentation_params'].append({
                    'original_file': img_file.name,
                    'augmented_file': output_filename,
                    'aug_id': aug_id,
                    'params': params
                })

                stats['total_augmented'] += 1

                # 统计颜色增强使用情况
                if params['use_color_aug']:
                    stats['color_augmented'] += 1
                else:
                    stats['geometric_only'] += 1

        # 保存统计信息
        stats_file = output_path / 'augmentation_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"\n增强完成!")
        print(f"原始图片: {stats['total_original']}")
        print(f"增强图片: {stats['total_augmented']}")
        print(f"\n增强策略分布:")
        print(f"  仅几何变换: {stats['geometric_only']} 张 ({stats['geometric_only']/stats['total_augmented']*100:.1f}%)")
        print(f"  几何+颜色变换: {stats['color_augmented']} 张 ({stats['color_augmented']/stats['total_augmented']*100:.1f}%)")
        print(f"\n统计信息已保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='数据增强脚本')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='原始图片目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--num-augmentations', type=int, default=33,
                        help='每张图片生成的增强数量 (默认33)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认42)')

    args = parser.parse_args()

    # 创建增强器
    augmentor = DataAugmentor(seed=args.seed)

    # 处理数据集
    augmentor.process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations
    )


if __name__ == '__main__':
    main()
