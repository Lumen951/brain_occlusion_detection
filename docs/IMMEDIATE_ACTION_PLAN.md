# 立即行动计划 (Phase 2)

**目标**: 在2-3周内将模型性能从~50%提升到>70%

---

## 当前问题

❌ ViT-B/16: 52% 平均准确率 (预期: 85%+)
❌ ResNet-50: 46% 平均准确率 (预期: 70%+)
❌ 性能差距: 30-40个百分点

**根本原因**:
1. 数据集太小 (~5000-7500训练样本)
2. 训练策略不当 (freeze_backbone=True)
3. 预训练特征不适合遮挡任务

---

## Week 1: 训练策略优化

### Day 1-2: 数据诊断

**创建诊断脚本**:

```python
# scripts/analysis/diagnose_dataset.py
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def diagnose_dataset():
    """诊断数据集分布和潜在问题"""

    results = {}

    for split in ['train', 'val', 'test']:
        split_dir = Path(f'data/{split}')
        images = list(split_dir.glob('*.jpg'))

        # 解析文件名: Aircraft1_70%_2.jpg
        classes = []
        occlusions = []

        for img in images:
            parts = img.stem.split('_')
            classes.append(parts[0])
            occlusions.append(parts[1])

        results[split] = {
            'total': len(images),
            'class_dist': Counter(classes),
            'occlusion_dist': Counter(occlusions),
            'images': [img.name for img in images]
        }

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, split in enumerate(['train', 'val', 'test']):
        # 类别分布
        ax = axes[0, idx]
        class_counts = results[split]['class_dist']
        ax.bar(class_counts.keys(), class_counts.values())
        ax.set_title(f'{split.upper()} - Class Distribution')
        ax.set_ylabel('Count')

        # 遮挡分布
        ax = axes[1, idx]
        occ_counts = results[split]['occlusion_dist']
        ax.bar(occ_counts.keys(), occ_counts.values())
        ax.set_title(f'{split.upper()} - Occlusion Distribution')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('experiments/analysis/dataset_diagnosis.png', dpi=300)

    # 保存JSON
    with open('experiments/analysis/dataset_diagnosis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✅ 数据诊断完成")
    print(f"训练集: {results['train']['total']} 样本")
    print(f"验证集: {results['val']['total']} 样本")
    print(f"测试集: {results['test']['total']} 样本")

    return results

if __name__ == "__main__":
    diagnose_dataset()
```

**运行**:
```bash
python scripts/analysis/diagnose_dataset.py
```

### Day 3-5: 三个训练实验

**实验1: 全模型微调**

创建配置 `configs/vit_b16_unfreeze.yaml`:

```yaml
experiment:
  name: "vit_b16_unfreeze"
  phase: "phase2_improvement"

dataset:
  type: "image_split"
  train_dir: "data/train"
  val_dir: "data/val"
  test_dir: "data/test"
  image_size: 224
  batch_size: 16

model:
  type: "vit_b16"
  num_classes: 2
  pretrained: true
  freeze_backbone: false  # 关键改变
  drop_rate: 0.3

training:
  epochs: 100
  optimizer:
    type: "adamw"
    lr: 1.0e-5  # 降低学习率
    weight_decay: 0.05
  scheduler:
    type: "cosine"
    warmup_epochs: 10
  early_stopping:
    enabled: true
    patience: 15
```

**运行**:
```bash
python scripts/training/train_model.py --config configs/vit_b16_unfreeze.yaml
```

**实验2: 强数据增强**

创建配置 `configs/vit_b16_strong_aug.yaml`:

```yaml
# 基础配置同上,添加:

training:
  # ... 其他配置

  augmentation:
    # 基础增强
    random_horizontal_flip: true
    random_rotation: 30
    color_jitter:
      brightness: 0.3
      contrast: 0.3
      saturation: 0.2
      hue: 0.1

    # 高级增强
    random_erasing:
      p: 0.3
      scale: [0.02, 0.33]
      ratio: [0.3, 3.3]

    # Mixup/CutMix
    mixup_alpha: 0.2
    cutmix_alpha: 1.0
    cutmix_prob: 0.5
```

**运行**:
```bash
python scripts/training/train_model.py --config configs/vit_b16_strong_aug.yaml
```

**实验3: ResNet-50 全模型微调**

创建配置 `configs/resnet50_unfreeze.yaml`:

```yaml
experiment:
  name: "resnet50_unfreeze"
  phase: "phase2_improvement"

dataset:
  type: "image_split"
  train_dir: "data/train"
  val_dir: "data/val"
  test_dir: "data/test"
  image_size: 224
  batch_size: 32  # ResNet可以用更大batch

model:
  type: "resnet50"
  num_classes: 2
  pretrained: true
  freeze_backbone: false
  drop_rate: 0.5

training:
  epochs: 100
  optimizer:
    type: "adamw"
    lr: 3.0e-5  # ResNet可以用稍高学习率
    weight_decay: 0.1
  scheduler:
    type: "cosine"
    warmup_epochs: 5
  early_stopping:
    enabled: true
    patience: 15
```

**运行**:
```bash
python scripts/training/train_model.py --config configs/resnet50_unfreeze.yaml
```

### Day 6-7: 结果对比

**创建对比脚本** `scripts/analysis/compare_experiments.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compare_experiments():
    """对比三个实验的结果"""

    experiments = {
        'ViT Unfreeze': 'experiments/vit_b16_unfreeze/metrics',
        'ViT Strong Aug': 'experiments/vit_b16_strong_aug/metrics',
        'ResNet Unfreeze': 'experiments/resnet50_unfreeze/metrics'
    }

    results = {}

    for name, path in experiments.items():
        metrics_file = Path(path) / 'test_metrics_by_occlusion.csv'
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            results[name] = df

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    occlusion_levels = ['10%', '70%', '90%']

    for idx, occ in enumerate(occlusion_levels):
        ax = axes[idx]

        for name, df in results.items():
            row = df[df['occlusion_level'] == occ]
            if not row.empty:
                acc = row['accuracy'].values[0]
                ax.bar(name, acc, label=name)

        ax.set_title(f'{occ} Occlusion')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.7, color='r', linestyle='--', label='Target (70%)')

    plt.tight_layout()
    plt.savefig('experiments/analysis/experiment_comparison.png', dpi=300)

    # 打印最佳结果
    print("\n" + "="*50)
    print("实验结果对比")
    print("="*50)

    for name, df in results.items():
        avg_acc = df['accuracy'].mean()
        print(f"\n{name}:")
        print(f"  平均准确率: {avg_acc:.2%}")
        for _, row in df.iterrows():
            print(f"  {row['occlusion_level']}: {row['accuracy']:.2%}")

    print("\n" + "="*50)

if __name__ == "__main__":
    compare_experiments()
```

**运行**:
```bash
python scripts/analysis/compare_experiments.py
```

---

## Week 2: MAE-ViT 实验

### Day 1-3: MAE-ViT 训练

**创建配置** `configs/mae_vit_base.yaml`:

```yaml
experiment:
  name: "mae_vit_base"
  phase: "phase2_improvement"

dataset:
  type: "image_split"
  train_dir: "data/train"
  val_dir: "data/val"
  test_dir: "data/test"
  image_size: 224
  batch_size: 16

model:
  type: "mae_vit_base"  # 需要在pretrained_loader.py中添加
  num_classes: 2
  pretrained: true
  pretrained_path: "mae_pretrain_vit_base.pth"  # 下载MAE权重
  freeze_backbone: false
  drop_rate: 0.3

training:
  epochs: 100
  optimizer:
    type: "adamw"
    lr: 5.0e-6  # MAE通常需要更低学习率
    weight_decay: 0.05
  scheduler:
    type: "cosine"
    warmup_epochs: 10
  early_stopping:
    enabled: true
    patience: 15
```

**修改** `src/models/pretrained_loader.py`:

```python
def create_mae_vit_base_pretrained(num_classes=2, pretrained=True,
                                    pretrained_path=None, **kwargs):
    """
    创建MAE预训练的ViT-Base模型

    Args:
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        pretrained_path: MAE预训练权重路径
    """
    import torch

    # 创建ViT-Base模型
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,  # 先不加载ImageNet权重
        num_classes=num_classes,
        **kwargs
    )

    if pretrained and pretrained_path:
        # 加载MAE预训练权重
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # MAE权重需要特殊处理
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 移除decoder相关的权重
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('decoder'):
                continue
            encoder_state_dict[k] = v

        # 加载权重 (忽略分类头)
        msg = model.load_state_dict(encoder_state_dict, strict=False)
        print(f"MAE权重加载: {msg}")

    return model
```

**下载MAE权重**:
```bash
# 从 https://github.com/facebookresearch/mae 下载
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
mv mae_pretrain_vit_base.pth checkpoints/
```

**运行训练**:
```bash
python scripts/training/train_model.py --config configs/mae_vit_base.yaml
```

### Day 4-5: 集成学习

**创建集成脚本** `scripts/evaluation/ensemble_models.py`:

```python
import torch
import numpy as np
from pathlib import Path

def ensemble_predict(models, dataloader, method='soft_voting'):
    """
    集成多个模型的预测

    Args:
        models: 模型列表
        dataloader: 数据加载器
        method: 'soft_voting' 或 'hard_voting'
    """
    all_preds = []
    all_labels = []

    for model in models:
        model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()

            # 收集所有模型的预测
            batch_preds = []
            for model in models:
                outputs = model(images)
                if method == 'soft_voting':
                    probs = torch.softmax(outputs, dim=1)
                    batch_preds.append(probs.cpu().numpy())
                else:
                    preds = torch.argmax(outputs, dim=1)
                    batch_preds.append(preds.cpu().numpy())

            # 集成
            if method == 'soft_voting':
                # 平均概率
                avg_probs = np.mean(batch_preds, axis=0)
                ensemble_pred = np.argmax(avg_probs, axis=1)
            else:
                # 投票
                votes = np.array(batch_preds)
                ensemble_pred = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(),
                    axis=0,
                    arr=votes
                )

            all_preds.extend(ensemble_pred)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

def main():
    # 加载最佳的3个模型
    models = []

    checkpoints = [
        'experiments/vit_b16_unfreeze/checkpoints/best_model.pth',
        'experiments/resnet50_unfreeze/checkpoints/best_model.pth',
        'experiments/mae_vit_base/checkpoints/best_model.pth'
    ]

    for ckpt in checkpoints:
        if Path(ckpt).exists():
            model = load_model(ckpt)
            models.append(model)

    # 评估集成
    test_loader = create_test_loader()
    preds, labels = ensemble_predict(models, test_loader, method='soft_voting')

    # 计算准确率
    accuracy = (preds == labels).mean()
    print(f"集成准确率: {accuracy:.2%}")

    # 按遮挡等级分析
    # ...

if __name__ == "__main__":
    main()
```

### Day 6-7: 错误分析与可视化

**创建attention可视化脚本** `scripts/visualization/visualize_attention.py`:

```python
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_attention_maps(model, image_path, save_dir):
    """
    可视化ViT的attention maps

    Args:
        model: ViT模型
        image_path: 图像路径
        save_dir: 保存目录
    """
    from PIL import Image
    import torchvision.transforms as transforms

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).cuda()

    # 提取attention
    model.eval()
    with torch.no_grad():
        # 需要修改模型forward来返回attention
        outputs, attentions = model(input_tensor, return_attention=True)

    # 可视化最后一层的attention
    last_attn = attentions[-1]  # [1, num_heads, num_patches, num_patches]

    # 平均所有head
    avg_attn = last_attn.mean(dim=1)[0]  # [num_patches, num_patches]

    # 取CLS token对所有patch的attention
    cls_attn = avg_attn[0, 1:]  # 忽略CLS自己

    # Reshape到2D
    num_patches = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(num_patches, num_patches)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(attn_map.cpu().numpy(), cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / f'{Path(image_path).stem}_attention.png', dpi=300)

if __name__ == "__main__":
    # 可视化错误样本的attention
    pass
```

---

## Week 3: 决策与下一步

### 评估标准

**成功** (进入Phase 3):
- ✅ 至少一个模型 >70% 平均准确率
- ✅ 性能曲线符合预期 (10% > 70% > 90%)
- ✅ 错误分析显示有意义的模式

**部分成功** (调整Phase 3):
- ⚠️ 模型达到 60-70% 准确率
- ⚠️ 降低Phase 3的模型数量 (3-4个)
- ⚠️ 聚焦于方法论贡献

**失败** (转向):
- ❌ 模型 <60% 准确率
- ❌ 转向小数据集训练策略研究
- ❌ 或扩充数据集后重新开始

### 决策树

```
Week 3 结束评估:
│
├─ 最佳模型 >70%?
│  ├─ 是 → 进入Phase 3 (5-7个模型对比)
│  └─ 否 → 继续下面判断
│
├─ 最佳模型 60-70%?
│  ├─ 是 → 调整Phase 3 (3-4个模型,降低目标)
│  └─ 否 → 继续下面判断
│
└─ 最佳模型 <60%?
   ├─ 数据扩充可行? → 扩充数据,重新训练
   └─ 数据扩充不可行 → 转向方法论研究
```

---

## 需要创建的文件清单

**配置文件**:
- [ ] `configs/vit_b16_unfreeze.yaml`
- [ ] `configs/vit_b16_strong_aug.yaml`
- [ ] `configs/resnet50_unfreeze.yaml`
- [ ] `configs/mae_vit_base.yaml`

**脚本文件**:
- [ ] `scripts/analysis/diagnose_dataset.py`
- [ ] `scripts/analysis/compare_experiments.py`
- [ ] `scripts/evaluation/ensemble_models.py`
- [ ] `scripts/visualization/visualize_attention.py`

**修改文件**:
- [ ] `src/models/pretrained_loader.py` (添加MAE支持)
- [ ] `scripts/training/train_model.py` (支持新的augmentation配置)

---

## 检查清单

**Week 1**:
- [ ] Day 1-2: 数据诊断完成
- [ ] Day 3-5: 三个实验启动
- [ ] Day 6-7: 结果对比完成

**Week 2**:
- [ ] Day 1-3: MAE-ViT训练完成
- [ ] Day 4-5: 集成学习测试
- [ ] Day 6-7: 错误分析和可视化

**Week 3**:
- [ ] 评估所有实验结果
- [ ] 决策: 进入Phase 3 或调整方向
- [ ] 与导师沟通下一步计划

---

**创建日期**: 2026-01-08
**预计完成**: 2026-01-29 (3周)
**负责人**: [你的名字]
