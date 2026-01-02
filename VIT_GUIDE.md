# ViT模型训练指南 - 遮挡飞机图像分类

## 数据集架构分析

### 数据集信息
- **名称**: ds005226 - Occluded Image Interpretation Dataset (OIID)
- **任务**: 二分类（Aircraft1 vs Aircraft2）
- **数据量**: 65名受试者，每人观看300张刺激图片
- **图片类别**: 2类飞机，每类150张
- **遮挡等级**: 10%, 70%, 90%三个等级
- **图片格式**: JPEG (位于 `stimuli/` 文件夹)
- **标签文件**: TSV格式的事件文件 (`*_events.tsv`)

### 数据集结构
```
E:\Dataset\ds005226\
├── stimuli/                          # 刺激图片文件夹
│   ├── Aircraft1_10%_*.jpg          # 飞机1，10%遮挡
│   ├── Aircraft1_70%_*.jpg          # 飞机1，70%遮挡
│   ├── Aircraft1_90%_*.jpg          # 飞机1，90%遮挡
│   ├── Aircraft2_10%_*.jpg          # 飞机2，10%遮挡
│   └── ...
├── sub-01/                          # 受试者1
│   └── ses-01/func/                 # 功能扫描数据
│       └── *_task-image_run-*_events.tsv  # 事件标签文件
└── sub-02/ ... sub-65/              # 其他受试者
```

## 已创建的文件

### 1. 数据集类 (`src/data/stimulus_dataset.py`)
- `OccludedAircraftDataset`: 用于加载刺激图片和标签的Dataset类
- `create_dataloaders()`: 便捷函数，创建train/val/test DataLoader
- 支持按遮挡等级过滤数据
- 自动从TSV事件文件中提取标签

### 2. ViT模型 (`src/models/vit_model.py`)
提供4种预设模型配置：
- **ViT-Tiny**: ~2M参数，快速训练，适合初步实验
- **ViT-Small**: ~22M参数，性能与速度平衡（推荐）
- **ViT-Base**: ~86M参数，标准ViT架构
- **ViT-Large**: ~304M参数，最高精度，训练较慢

### 3. 训练配置 (`configs/vit_config.yaml`)
完整的训练配置，包括：
- 数据集路径和受试者划分（50训练/8验证/7测试）
- 模型架构选择
- 训练超参数（学习率、优化器、scheduler等）
- 日志和检查点设置

### 4. 训练脚本 (`train_vit.py`)
完整的训练循环实现：
- 混合精度训练（AMP）支持
- TensorBoard日志
- 早停机制
- 模型检查点保存
- 验证集评估

## 配置建议

### 方案1: 快速实验 (推荐初期使用)
```yaml
model:
  size: "small"  # ViT-Small

training:
  batch_size: 32
  epochs: 50
  optimizer:
    lr: 3.0e-4
  use_amp: true
```

**优点**: 训练快，资源消耗少，适合超参数调优

### 方案2: 高性能配置
```yaml
model:
  size: "base"  # ViT-Base

training:
  batch_size: 16  # 如果显存不足，可降至8
  epochs: 100
  optimizer:
    lr: 1.0e-4  # Base模型用更小的学习率
  use_amp: true
```

**优点**: 更好的性能，适合最终模型训练

### 方案3: 按遮挡等级训练
如果想研究不同遮挡等级的影响，可以在配置中设置：
```yaml
dataset:
  occlusion_levels: [0.1]  # 只训练10%遮挡的图片
```

## 使用步骤

### 1. 安装依赖 (正在进行中)
```bash
uv add vit-pytorch pillow pandas
```

### 2. 验证数据集路径
确认配置文件中的路径正确：
```yaml
dataset:
  root: "E:/Dataset/ds005226"  # 确保路径存在
```

### 3. 开始训练
```bash
# 使用默认配置
python train_vit.py

# 使用自定义配置
python train_vit.py --config configs/my_config.yaml
```

### 4. 监控训练
```bash
# 启动TensorBoard
tensorboard --logdir experiments/logs
```

在浏览器中打开 http://localhost:6006 查看训练曲线

## 预期结果

### 数据量统计
- 每个受试者约有100-150个有效样本（排除rest trials）
- 50个训练受试者 ≈ 5000-7500训练样本
- 8个验证受试者 ≈ 800-1200验证样本
- 7个测试受试者 ≈ 700-1050测试样本

### 性能预期
- **基线准确率**: ~50%（随机猜测）
- **10%遮挡**: 预期>90%准确率（接近无遮挡）
- **70%遮挡**: 预期70-85%准确率
- **90%遮挡**: 预期50-70%准确率（高度遮挡，更具挑战）

## 进阶配置

### 1. 数据增强调整
在 `src/data/stimulus_dataset.py` 的 `get_default_transforms()` 中修改：
```python
transforms.RandomHorizontalFlip(p=0.5),  # 调整翻转概率
transforms.RandomRotation(degrees=15),    # 调整旋转角度
transforms.ColorJitter(...)               # 调整颜色抖动
```

### 2. 学习率调优
不同模型大小建议不同学习率：
- ViT-Tiny/Small: 3e-4 到 5e-4
- ViT-Base: 1e-4 到 3e-4
- ViT-Large: 5e-5 到 1e-4

### 3. 批次大小与显存
如果遇到OOM（Out of Memory）错误：
- 减小 `batch_size`
- 使用更小的模型 (`tiny` or `small`)
- 降低 `image_size` (224 → 128)

### 4. 分析不同遮挡等级
训练后，可以使用以下代码分析各遮挡等级的性能：
```python
# TODO: 可以添加评估脚本来分层分析各遮挡等级的性能
```

## 常见问题

### Q1: 数据集路径找不到
确保配置文件中的路径使用正确的格式（Windows用正斜杠或双反斜杠）：
```yaml
root: "E:/Dataset/ds005226"  # 推荐
# 或
root: "E:\\Dataset\\ds005226"
```

### Q2: 训练速度慢
- 启用混合精度训练: `use_amp: true`
- 增加 `num_workers` (但不要超过CPU核心数)
- 使用更小的模型

### Q3: 验证集准确率不提升
- 检查是否过拟合（训练准确率远高于验证准确率）
- 增加weight_decay
- 添加dropout
- 减少模型大小

## 下一步建议

1. **先运行小规模实验**: 用少量受试者（5-10个）快速验证代码是否正常工作
2. **调整受试者划分**: 根据需要修改train/val/test的受试者分配
3. **添加评估脚本**: 创建独立的评估脚本用于测试集评估和可视化
4. **实验追踪**: 使用wandb或mlflow进行实验管理
