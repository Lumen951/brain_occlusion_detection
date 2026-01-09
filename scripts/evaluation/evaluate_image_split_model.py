"""
评估image_split模式的模型在测试集上的性能

功能：
1. 从data/test/目录加载测试图像
2. 使用训练好的模型进行预测
3. 保存详细的预测结果（包括置信度）
4. 按遮挡级别统计性能
5. 生成与人类数据对比的CSV

用法:
python scripts/evaluate_image_split_model.py --model vit_b16
python scripts/evaluate_image_split_model.py --model resnet50
"""

import sys
from pathlib import Path
import argparse
import re
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.pretrained_loader import create_vit_b16_pretrained, create_resnet50_pretrained


class ImageSplitTestDataset:
    """简单的测试集数据加载器，用于image_split模式"""

    def __init__(self, test_dir: str, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform

        # 获取所有图像
        self.images = sorted(list(self.test_dir.glob("*.jpg")))

        if len(self.images) == 0:
            raise ValueError(f"No images found in {test_dir}")

        print(f"Found {len(self.images)} test images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # 从文件名提取信息
        # Aircraft1_10%_16.jpg -> (Aircraft1, 0.1, 16, 0)
        info = self._parse_filename(img_path.name)

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, info['label'], info

    def _parse_filename(self, filename):
        """解析文件名，提取标签和元数据"""
        # Aircraft1_10%_16.jpg
        parts = filename.replace('.jpg', '').split('_')

        aircraft_type = parts[0]  # Aircraft1 or Aircraft2
        occlusion_str = parts[1]  # 10%, 70%, or 90%
        image_num = parts[2] if len(parts) > 2 else 0

        # 提取标签
        label = 0 if aircraft_type == 'Aircraft1' else 1

        # 提取遮挡级别
        occlusion_match = re.search(r'(\d+)%', occlusion_str)
        if occlusion_match:
            occlusion_level = int(occlusion_match.group(1)) / 100.0
        else:
            occlusion_level = 0.1

        return {
            'filename': filename,
            'aircraft_type': aircraft_type,
            'occlusion_level': occlusion_level,
            'image_num': image_num,
            'label': label,
            'occlusion_str': occlusion_str
        }


def get_transforms(img_size=224):
    """获取图像变换"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_model(model_type, checkpoint_path, device):
    """加载训练好的模型"""
    print(f"Loading {model_type} model from {checkpoint_path}")

    # 创建模型
    if 'resnet' in model_type.lower():
        model = create_resnet50_pretrained(num_classes=2, pretrained=False)
    else:
        model = create_vit_b16_pretrained(num_classes=2, pretrained=False)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded")
    return model


def evaluate_model(model, dataset, device):
    """在测试集上评估模型"""
    results = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            image, label, info = dataset[i]

            # 添加batch维度
            image = image.unsqueeze(0).to(device)

            # 预测
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

            results.append({
                'image_name': info['filename'],
                'ground_truth': label,
                'predicted_class': pred.item(),
                'confidence': confidence.item(),
                'prob_class_0': probs[0, 0].item(),
                'prob_class_1': probs[0, 1].item(),
                'correct': (pred.item() == label),
                'occlusion_level': info['occlusion_level'],
                'occlusion_str': info['occlusion_str'],
                'aircraft_type': info['aircraft_type'],
                'label': label
            })

    return pd.DataFrame(results)


def calculate_metrics_by_occlusion(df):
    """按遮挡级别计算指标"""
    metrics = []

    for occ_level in sorted(df['occlusion_level'].unique()):
        subset = df[df['occlusion_level'] == occ_level]

        acc = (subset['correct'].sum() / len(subset))

        # 计算precision, recall, f1
        tp = ((subset['predicted_class'] == 1) & (subset['ground_truth'] == 1)).sum()
        fp = ((subset['predicted_class'] == 1) & (subset['ground_truth'] == 0)).sum()
        fn = ((subset['predicted_class'] == 0) & (subset['ground_truth'] == 1)).sum()
        tn = ((subset['predicted_class'] == 0) & (subset['ground_truth'] == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics.append({
            'occlusion_level': f"{int(occ_level*100)}%",
            'num_samples': len(subset),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct': int(subset['correct'].sum())
        })

    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on image_split test set')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vit_b16', 'resnet50'],
                       help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: auto-detect)')
    parser.add_argument('--test_dir', type=str, default='data/test',
                       help='Test directory')
    parser.add_argument('--output_dir', type=str, default='experiments/analysis',
                       help='Output directory')
    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 自动检测checkpoint路径
    if args.checkpoint is None:
        if args.model == 'vit_b16':
            args.checkpoint = 'experiments/vit_b16/image_split/checkpoints/best_model.pth'
        else:
            args.checkpoint = 'experiments/resnet50/image_split/checkpoints/best_model.pth'

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 加载模型
    model = load_model(args.model, checkpoint_path, device)

    # 创建测试数据集
    transform = get_transforms(img_size=224)
    test_dir = project_root / args.test_dir

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    dataset = ImageSplitTestDataset(str(test_dir), transform=transform)

    # 评估
    results_df = evaluate_model(model, dataset, device)

    # 计算指标
    overall_acc = (results_df['correct'].sum() / len(results_df))
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({results_df['correct'].sum()}/{len(results_df)})")

    # 按遮挡级别统计
    metrics_df = calculate_metrics_by_occlusion(results_df)

    print("\n" + "="*80)
    print("Performance by Occlusion Level")
    print("="*80)
    print(metrics_df.to_string(index=False))

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细预测结果
    predictions_file = output_dir / f"{args.model}_test_predictions.csv"
    results_df.to_csv(predictions_file, index=False)
    print(f"\n[OK] Saved predictions to: {predictions_file}")

    # 保存按遮挡级别的统计
    metrics_file = output_dir / f"{args.model}_metrics_by_occlusion.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"[OK] Saved metrics to: {metrics_file}")

    # 打印汇总
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Test set: {len(results_df)} images")
    print(f"Overall accuracy: {overall_acc:.2%}")
    print("\nBy occlusion level:")
    for _, row in metrics_df.iterrows():
        print(f"  {row['occlusion_level']}: {row['accuracy']:.2%} "
              f"({row['correct']}/{row['num_samples']})")


if __name__ == '__main__':
    main()
