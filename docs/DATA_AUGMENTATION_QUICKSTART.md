# 数据扩充快速开始指南

**目标**: 从300张原始图片生成10,000张训练图片

---

## 📋 准备工作

### 1. 确认环境

```bash
# 检查Python版本 (需要3.7+)
python --version

# 安装依赖
pip install pillow numpy tqdm
```

### 2. 确认路径

**原始图片**: `e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original`
- 应该包含300张图片
- 文件名格式: `Aircraft{1|2}_{10|70|90}%_{id}_original.jpg`

**输出目录**: `d:\University\Junior\1st\code\brain_occlusion_detection\data`

---

## 🚀 执行步骤

### Step 1: 数据增强 (生成~10,000张增强图片)

```bash
cd d:\University\Junior\1st\code\brain_occlusion_detection

python scripts/data_preparation/augment_dataset.py \
  --input-dir "e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original" \
  --output-dir "data/augmented_images" \
  --num-augmentations 33 \
  --seed 42
```

**预计时间**: 10-15分钟
**输出**: `data/augmented_images/` (约9,900张图片)

**检查点**:
```bash
# 检查生成的图片数量
ls data/augmented_images | wc -l
# 应该显示: 9900 (300 × 33)

# 查看统计信息
cat data/augmented_images/augmentation_stats.json
```

### Step 2: 添加遮挡 (为增强图片添加随机遮挡)

```bash
python scripts/data_preparation/add_occlusion.py \
  --input-dir "data/augmented_images" \
  --output-dir "data/train_augmented" \
  --mask-size 10 \
  --seed 42
```

**预计时间**: 15-20分钟
**输出**: `data/train_augmented/` (约9,900张带遮挡的图片)

**检查点**:
```bash
# 检查生成的图片数量
ls data/train_augmented | wc -l
# 应该显示: 9900

# 查看遮挡统计
cat data/train_augmented/occlusion_stats.json
```

### Step 3: 划分数据集 (训练集 vs 验证集)

```bash
python scripts/data_preparation/split_dataset.py \
  --original-dir "e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original" \
  --augmented-dir "data/train_augmented" \
  --train-dir "data/train" \
  --val-dir "data/val"
```

**预计时间**: 2-3分钟
**输出**:
- `data/train/` (9,900张增强图片)
- `data/val/` (300张原始图片)

**检查点**:
```bash
# 检查训练集
ls data/train | wc -l
# 应该显示: 9900

# 检查验证集
ls data/val | wc -l
# 应该显示: 300

# 查看数据集统计
cat data/dataset_split_stats.json
```

---

## ✅ 验证结果

### 1. 检查数据集平衡性

```python
import json

# 读取统计信息
with open('data/dataset_split_stats.json', 'r') as f:
    stats = json.load(f)

# 训练集
print("训练集:")
print(f"  总计: {stats['train']['total']}")
print(f"  Aircraft1: {stats['train']['by_class']['Aircraft1']}")
print(f"  Aircraft2: {stats['train']['by_class']['Aircraft2']}")
print(f"  10%遮挡: {stats['train']['by_occlusion']['10%']}")
print(f"  70%遮挡: {stats['train']['by_occlusion']['70%']}")
print(f"  90%遮挡: {stats['train']['by_occlusion']['90%']}")

# 验证集
print("\n验证集:")
print(f"  总计: {stats['val']['total']}")
print(f"  Aircraft1: {stats['val']['by_class']['Aircraft1']}")
print(f"  Aircraft2: {stats['val']['by_class']['Aircraft2']}")
print(f"  10%遮挡: {stats['val']['by_occlusion']['10%']}")
print(f"  70%遮挡: {stats['val']['by_occlusion']['70%']}")
print(f"  90%遮挡: {stats['val']['by_occlusion']['90%']}")
```

**预期输出**:
```
训练集:
  总计: 9900
  Aircraft1: 4950
  Aircraft2: 4950
  10%遮挡: 3300
  70%遮挡: 3300
  90%遮挡: 3300

验证集:
  总计: 300
  Aircraft1: 150
  Aircraft2: 150
  10%遮挡: 100
  70%遮挡: 100
  90%遮挡: 100
```

### 2. 可视化检查

```python
from PIL import Image
import matplotlib.pyplot as plt

# 随机选择几张图片查看
import random

train_images = list(Path('data/train').glob('*.jpg'))
val_images = list(Path('data/val').glob('*.jpg'))

# 显示训练集样本
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('训练集样本 (增强+遮挡)')

for i, ax in enumerate(axes.flat):
    img_path = random.choice(train_images)
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(img_path.name, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('data/train_samples.png', dpi=150)
print("训练集样本已保存到: data/train_samples.png")

# 显示验证集样本
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('验证集样本 (原始图片)')

for i, ax in enumerate(axes.flat):
    img_path = random.choice(val_images)
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(img_path.name, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('data/val_samples.png', dpi=150)
print("验证集样本已保存到: data/val_samples.png")
```

---

## 🔧 故障排除

### 问题1: 找不到原始图片

**错误**: `找到 0 张原始图片`

**解决**:
```bash
# 检查路径是否正确
ls "e:\Dataset\ds005226\derivatives\stimuli_dataset\stimuli_original" | head -5

# 如果路径不对,修改命令中的 --input-dir 参数
```

### 问题2: 内存不足

**错误**: `MemoryError`

**解决**:
```bash
# 减少每张图片的增强数量
python scripts/data_preparation/augment_dataset.py \
  --num-augmentations 20  # 从33减少到20
```

### 问题3: 生成速度太慢

**优化**:
- 关闭其他程序释放内存
- 使用SSD存储输出文件
- 减少增强数量

### 问题4: 图片质量下降

**检查**:
```python
# 对比原始图片和增强图片
from PIL import Image

original = Image.open('e:/Dataset/ds005226/derivatives/stimuli_dataset/stimuli_original/Aircraft1_10%_1_original.jpg')
augmented = Image.open('data/augmented_images/Aircraft1_10%_1_aug0.jpg')

# 显示对比
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original)
axes[0].set_title('Original')
axes[1].imshow(augmented)
axes[1].set_title('Augmented')
plt.show()
```

**调整**: 如果质量下降明显,修改 `augment_dataset.py` 中的变换参数

---

## 📊 预期效果

### 数据集规模对比

| 数据集 | 之前 | 之后 | 增长 |
|--------|------|------|------|
| 训练集 | 210 | 9,900 | 47× |
| 验证集 | 42 | 300 | 7× |
| 总计 | 252 | 10,200 | 40× |

### 预期性能提升

基于文献和经验:

| 模型 | 当前性能 | 预期性能 | 提升 |
|------|----------|----------|------|
| ViT-B/16 | 52% | 70-80% | +18-28% |
| ResNet-50 | 46% | 65-75% | +19-29% |
| MAE-ViT | - | 75-85% | 新模型 |

---

## 🎯 下一步

数据集生成完成后:

### 1. 更新配置文件

确保训练配置指向新的数据集:

```yaml
# configs/vit_b16_image_split.yaml
dataset:
  type: "image_split"
  train_dir: "data/train"  # 9,900张增强图片
  val_dir: "data/val"      # 300张原始图片
  test_dir: ""             # 不使用测试集
```

### 2. 重新训练模型

```bash
# ViT-B/16
python scripts/training/train_model.py --config configs/vit_b16_image_split.yaml

# ResNet-50
python scripts/training/train_model.py --config configs/resnet50_image_split.yaml

# MAE-ViT (如果已实现)
python scripts/training/train_model.py --config configs/mae_vit_base.yaml
```

### 3. 评估性能

```bash
# 评估模型
python scripts/evaluation/evaluate_by_occlusion.py \
  --checkpoint experiments/vit_b16/image_split/checkpoints/best_model.pth \
  --config configs/vit_b16_image_split.yaml

# 对比新旧数据集的效果
python scripts/analysis/compare_experiments.py
```

---

## 📞 需要帮助?

如果遇到问题:
1. 检查 `data/augmented_images/augmentation_stats.json`
2. 检查 `data/train_augmented/occlusion_stats.json`
3. 检查 `data/dataset_split_stats.json`
4. 参考 `docs/DATA_AUGMENTATION_PLAN.md` 的详细说明

---

**创建日期**: 2026-01-08
**预计总时间**: 30-40分钟
**预计存储空间**: ~200MB
