# Phase 1 实验分析与下一阶段研究计划

**生成日期**: 2026-01-08
**基于**: 当前ViT-B/16和ResNet-50实验结果
**目标**: 制定科学的下一阶段研究策略

---

## 1. Phase 1 实验结果总结

### 1.1 模型性能分析

#### 定量结果

| 模型 | 10%遮挡 | 70%遮挡 | 90%遮挡 | 平均准确率 | 鲁棒性指标* |
|------|---------|---------|---------|------------|-------------|
| **ViT-B/16** | 50.0% | 50.0% | 56.25% | 52.08% | -6.25% |
| **ResNet-50** | 43.75% | 43.75% | 50.0% | 45.83% | -6.25% |
| **人类** | ~95%+ | ~80%+ | ~68%+ | ~81%+ | ~27% |

*鲁棒性指标 = 90%遮挡准确率 - 10%遮挡准确率 (越小越好)

#### 关键观察

**⚠️ 严重问题识别**:

1. **性能远低于预期**
   - 原计划预期: ViT 85-98%, ResNet 65-96%
   - 实际结果: ViT ~52%, ResNet ~46%
   - **差距**: 30-40个百分点

2. **异常的性能曲线**
   - ViT在90%遮挡下表现**优于**10%遮挡 (56.25% vs 50%)
   - 这违反直觉,可能表明:
     - 训练数据分布不均
     - 模型学习到了错误的特征
     - 测试集样本量太小(每个遮挡等级仅16个样本)

3. **人类vs AI差距巨大**
   - 人类准确率 > 80% (平均)
   - AI准确率 < 55% (平均)
   - **差距**: 25-35个百分点

### 1.2 错误模式分析

基于`per_image_comparison_summary.csv`的48个测试样本:

```
错误类型分布:
├─ 所有模型都正确: 4/48 (8.3%)
├─ 所有模型都错误: 0/48 (0%)
├─ 人类对但AI错: ~35/48 (72.9%)
├─ ViT对ResNet错: ~15/48 (31.3%)
└─ ResNet对ViT错: ~12/48 (25.0%)
```

**关键发现**:
- ViT和ResNet的错误样本重叠度约为 **50-60%**
- 两个模型在不同样本上犯错,显示出架构差异
- 但整体性能都太低,难以得出有意义的结论

### 1.3 根本原因分析

#### 可能的问题来源

**1. 数据集规模问题**
```
训练集: ~5000-7500样本 (50个受试者)
验证集: ~800-1200样本 (8个受试者)
测试集: ~700-1050样本 (7个受试者)
```
- 对于从头训练深度模型,数据量严重不足
- 即使使用预训练模型,fine-tuning也需要更多数据

**2. 训练策略问题**
- 当前配置: `freeze_backbone=True` (只训练分类头)
- 可训练参数: ResNet ~2K, ViT ~1.5K
- **问题**: 预训练的ImageNet特征可能不适合遮挡识别任务

**3. 数据不平衡问题**
- 二分类任务 (Aircraft1 vs Aircraft2)
- 可能存在类别不平衡
- 遮挡等级分布可能不均

**4. 评估指标问题**
- 测试集每个遮挡等级仅16个样本
- 统计显著性不足
- 需要更大的测试集或交叉验证

---

## 2. 文献调研关键发现

### 2.1 Vision Transformers与遮挡鲁棒性

**核心文献方向**:

1. **Masked Autoencoders (MAE)** [He et al., CVPR 2022]
   - 通过masked modeling预训练,天然适合遮挡场景
   - 在遮挡鲁棒性上显著优于标准ViT
   - **启示**: 应该优先测试MAE-ViT

2. **Attention机制与遮挡**
   - ViT的全局attention理论上更适合处理遮挡
   - 但需要足够的训练数据来学习这种能力
   - **启示**: 当前数据量可能不足以发挥ViT优势

3. **CNN vs Transformer对比研究**
   - 最新研究显示: 在小数据集上,CNN往往优于ViT
   - ViT需要更多数据才能超越CNN
   - **启示**: 当前结果符合文献预期

### 2.2 小数据集训练策略

**有效方法**:

1. **渐进式解冻 (Progressive Unfreezing)**
   ```python
   Stage 1: 只训练分类头 (10 epochs)
   Stage 2: 解冻最后2层 (10 epochs)
   Stage 3: 解冻最后4层 (10 epochs)
   Stage 4: 全模型微调 (20 epochs)
   ```

2. **强数据增强**
   - Mixup / CutMix
   - RandAugment
   - 模拟遮挡的增强

3. **正则化策略**
   - 更高的dropout (0.5-0.7)
   - Label smoothing
   - Stochastic depth

### 2.3 人类vs AI对比研究

**相关工作**:

1. **fMRI与深度学习对比**
   - 需要表征相似性分析 (RSA)
   - 需要足够的模型性能才能进行有意义的对比
   - **启示**: 当前模型性能太低,暂不适合fMRI分析

2. **错误模式分析**
   - 需要模型达到合理性能 (>70%)
   - 才能分析"为什么错"而不是"为什么都错"
   - **启示**: 需要先提升模型性能

---

## 3. 下一阶段研究计划 (修订版)

### 3.1 战略调整

**核心决策**:

❌ **不继续原计划的多模型对比**
✅ **先解决基础性能问题,再扩展**

**理由**:
1. 当前两个模型性能都太低 (~50%)
2. 在此基础上增加更多模型无意义
3. 需要先建立可靠的baseline

### 3.2 Phase 2 目标: 建立可靠Baseline

**时间**: 2-3周
**目标**: 将模型性能提升到 **>70%** (平均准确率)

#### 任务清单

**Week 1: 诊断与修复**

```python
Task 1.1: 数据分析 (2天)
├─ 检查类别平衡性
├─ 检查遮挡等级分布
├─ 可视化训练/验证/测试集分布
└─ 识别潜在的数据泄露

Task 1.2: 训练策略优化 (3天)
├─ 实验1: freeze_backbone=False (全模型微调)
├─ 实验2: 渐进式解冻
├─ 实验3: 更强的数据增强
└─ 对比三种策略的效果

Task 1.3: 超参数调优 (2天)
├─ 学习率搜索 (1e-5 to 1e-3)
├─ Weight decay调整
├─ Dropout rate调整
└─ 使用Optuna自动调优
```

**Week 2: MAE-ViT实验**

```python
Task 2.1: MAE-ViT训练 (3天)
├─ 加载MAE预训练权重
├─ Fine-tune on OIID
├─ 对比与标准ViT的差异
└─ 预期: MAE应该显著优于ViT

Task 2.2: 集成学习 (2天)
├─ ViT + ResNet + MAE集成
├─ 软投票 / 硬投票
└─ 预期: 集成应该 > 单模型

Task 2.3: 错误分析 (2天)
├─ 识别困难样本
├─ 可视化attention maps
└─ 为下一步提供线索
```

**Week 3: 数据扩充与验证**

```python
Task 3.1: 数据扩充 (3天)
├─ 方案A: 使用OIID的所有受试者数据
├─ 方案B: 添加人造遮挡数据 (COCO/ImageNet)
├─ 方案C: 数据增强策略优化
└─ 重新训练最佳模型

Task 3.2: 交叉验证 (2天)
├─ 5-fold交叉验证
├─ 评估模型稳定性
└─ 获得更可靠的性能估计

Task 3.3: 决策点 (2天)
├─ 评估: 是否达到>70%性能?
├─ 如果是 → 进入Phase 3 (多模型对比)
└─ 如果否 → 继续优化或调整方向
```

### 3.3 Phase 3: 多模型对比 (条件执行)

**前提**: Phase 2达到>70%性能

**时间**: 3-4周
**目标**: 系统对比5-7个模型

#### 模型选择 (优先级排序)

```
必选模型 (5个):
1. ViT-B/16 (已有,需重新训练)
2. ResNet-50 (已有,需重新训练)
3. MAE-ViT (Phase 2已训练)
4. Swin-B (层级Transformer)
5. ConvNeXt-B (现代CNN)

可选模型 (2-3个,根据时间):
6. DeiT-III (Data-efficient)
7. ResNet-101 (深层CNN)
8. CoAtNet (混合架构)
```

#### 分析维度

```
1. 性能对比
   ├─ 各遮挡等级准确率
   ├─ 鲁棒性指标
   └─ 统计显著性检验

2. 错误模式分析
   ├─ 错误样本重叠度 (Jaccard)
   ├─ 错误类型聚类
   └─ 典型错误样本可视化

3. Attention分析
   ├─ Attention entropy
   ├─ Attention maps可视化
   └─ 与人类fMRI对比 (如果性能足够)

4. 表征分析
   ├─ RSA (Representational Similarity Analysis)
   ├─ CCA (Canonical Correlation Analysis)
   └─ 表征稳定性分析
```

### 3.4 Phase 4: 论文撰写 (2-3周)

**根据Phase 2-3的结果,选择投稿策略**:

**场景A: Phase 2成功 + Phase 3完成**
```
投稿目标: AAAI 2026 / IJCAI 2026
论文类型: 多模型对比研究
贡献:
├─ 系统对比5-7个架构
├─ 错误模式深入分析
├─ Attention机制对比
└─ 为模型选择提供指导
```

**场景B: Phase 2成功 + Phase 3部分完成**
```
投稿目标: WACV 2026 / BMVC 2026
论文类型: 应用导向对比研究
贡献:
├─ 对比3-5个模型
├─ 聚焦遮挡鲁棒性
└─ 实用的模型选择建议
```

**场景C: Phase 2困难**
```
投稿目标: Workshop / ICPR 2026
论文类型: 方法论研究
贡献:
├─ 小数据集训练策略
├─ 遮挡识别的挑战分析
└─ 数据增强方法对比
```

---

## 4. 具体行动计划

### 4.1 立即行动 (本周)

**Day 1-2: 数据诊断**

```bash
# 创建数据分析脚本
python scripts/analysis/diagnose_dataset.py \
  --train-dir data/train \
  --val-dir data/val \
  --test-dir data/test \
  --output experiments/analysis/dataset_diagnosis.json

# 检查内容:
# 1. 类别分布
# 2. 遮挡等级分布
# 3. 样本数量统计
# 4. 潜在的数据问题
```

**Day 3-5: 训练策略实验**

```bash
# 实验1: 全模型微调
python scripts/training/train_model.py \
  --config configs/vit_b16_image_split_unfreeze.yaml

# 实验2: 渐进式解冻
python scripts/training/train_progressive.py \
  --config configs/vit_b16_progressive.yaml

# 实验3: 强数据增强
python scripts/training/train_model.py \
  --config configs/vit_b16_strong_aug.yaml
```

**Day 6-7: 结果对比与决策**

```bash
# 对比三个实验的结果
python scripts/analysis/compare_training_strategies.py \
  --exp1 experiments/vit_unfreeze \
  --exp2 experiments/vit_progressive \
  --exp3 experiments/vit_strong_aug

# 决策: 选择最佳策略用于后续实验
```

### 4.2 配置文件模板

**vit_b16_image_split_unfreeze.yaml**:

```yaml
experiment:
  name: "vit_b16_unfreeze"
  phase: "phase2_baseline_improvement"

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

  # 强数据增强
  augmentation:
    random_horizontal_flip: 0.5
    random_rotation: 30
    color_jitter:
      brightness: 0.3
      contrast: 0.3
      saturation: 0.2
    random_erasing: 0.2  # 模拟遮挡
    mixup_alpha: 0.2
    cutmix_alpha: 1.0
```

**vit_b16_progressive.yaml**:

```yaml
experiment:
  name: "vit_b16_progressive"
  phase: "phase2_baseline_improvement"

dataset:
  # ... 同上

model:
  type: "vit_b16"
  num_classes: 2
  pretrained: true
  drop_rate: 0.3

training:
  # 渐进式解冻策略
  progressive_unfreezing:
    enabled: true
    stages:
      - name: "head_only"
        epochs: 10
        freeze_backbone: true
        lr: 3.0e-4

      - name: "last_2_blocks"
        epochs: 10
        unfreeze_layers: ["blocks.10", "blocks.11"]
        lr: 1.0e-4

      - name: "last_4_blocks"
        epochs: 10
        unfreeze_layers: ["blocks.8", "blocks.9", "blocks.10", "blocks.11"]
        lr: 5.0e-5

      - name: "full_model"
        epochs: 20
        freeze_backbone: false
        lr: 1.0e-5
```

### 4.3 新增脚本需求

**scripts/analysis/diagnose_dataset.py**:

```python
"""
数据集诊断脚本
检查:
1. 类别分布
2. 遮挡等级分布
3. 样本数量
4. 图像质量
5. 潜在问题
"""

import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def diagnose_dataset(train_dir, val_dir, test_dir):
    results = {
        'train': analyze_split(train_dir),
        'val': analyze_split(val_dir),
        'test': analyze_split(test_dir)
    }

    # 生成诊断报告
    generate_report(results)

    return results

def analyze_split(split_dir):
    # 统计类别、遮挡等级、样本数
    # ...
    pass

if __name__ == "__main__":
    # 运行诊断
    pass
```

**scripts/training/train_progressive.py**:

```python
"""
渐进式解冻训练脚本
"""

def progressive_training(config):
    model = create_model(config)

    for stage in config['training']['progressive_unfreezing']['stages']:
        print(f"Stage: {stage['name']}")

        # 设置冻结/解冻
        set_freeze_layers(model, stage)

        # 训练
        train_stage(model, stage)

    return model
```

---

## 5. 风险评估与应对

### 5.1 主要风险

**风险1: Phase 2无法达到>70%性能**

```
概率: 中 (40%)
影响: 高 (无法继续多模型对比)

应对策略:
├─ Plan A: 扩充数据集 (使用更多OIID数据或人造数据)
├─ Plan B: 降低目标 (60%也可以接受,调整研究问题)
├─ Plan C: 转向方法论研究 (小数据集训练策略)
└─ Plan D: 使用预训练的遮挡鲁棒模型 (如MAE)
```

**风险2: 时间不足**

```
概率: 高 (60%)
影响: 中 (可能无法完成所有计划)

应对策略:
├─ 优先完成Phase 2 (必须)
├─ Phase 3根据时间灵活调整模型数量
├─ 准备多个投稿目标 (顶会/普通会议/workshop)
└─ 分阶段产出 (每个阶段都可独立成文)
```

**风险3: 导师不满意调整后的计划**

```
概率: 低 (20%)
影响: 高 (需要重新规划)

应对策略:
├─ 及早与导师沟通调整理由
├─ 展示当前结果和问题分析
├─ 说明调整后的计划更科学、更可行
└─ 强调分阶段产出的优势
```

### 5.2 成功标准 (修订)

**Phase 2 成功标准**:
```
✅ 至少一个模型达到>70%平均准确率
✅ 模型在各遮挡等级的性能符合预期趋势
✅ 错误分析显示有意义的模式
✅ 有足够的数据支持后续分析
```

**Phase 3 成功标准**:
```
✅ 完成3-5个模型的对比
✅ 发现2-3个有意义的架构差异
✅ 完成错误模式和attention分析
✅ 撰写完整的论文初稿
```

---

## 6. 预期成果

### 6.1 最坏情况

```
场景: Phase 2失败,无法提升性能

产出:
├─ 技术报告: "小数据集遮挡识别的挑战"
├─ 投稿目标: Workshop或技术报告
└─ 价值: 方法论贡献,负结果也有价值
```

### 6.2 中等情况

```
场景: Phase 2成功,Phase 3部分完成

产出:
├─ 论文: "ViT vs CNN在遮挡识别上的对比研究"
├─ 模型数量: 3-5个
├─ 投稿目标: WACV/BMVC/ICPR
└─ 价值: 扎实的对比研究,有实用价值
```

### 6.3 理想情况

```
场景: Phase 2-3都成功

产出:
├─ 论文: "多架构遮挡识别对比与分析"
├─ 模型数量: 5-7个
├─ 投稿目标: AAAI/IJCAI
└─ 价值: 系统的对比研究,理论与实践贡献
```

---

## 7. 总结与建议

### 7.1 核心建议

**1. 务实调整策略**
- 承认当前结果不理想
- 先解决基础问题,再扩展
- 分阶段产出,降低风险

**2. 聚焦关键问题**
- Phase 2: 提升模型性能到可用水平
- Phase 3: 在可靠baseline上进行对比
- Phase 4: 根据结果灵活选择投稿目标

**3. 保持灵活性**
- 每个阶段都有决策点
- 根据结果及时调整
- 准备多个备选方案

### 7.2 立即行动

**本周任务**:
1. ✅ 运行数据诊断脚本
2. ✅ 启动3个训练策略实验
3. ✅ 与导师沟通调整后的计划
4. ✅ 准备Week 2的MAE实验

**下周任务**:
1. 评估Week 1的实验结果
2. 选择最佳训练策略
3. 训练MAE-ViT
4. 决策: 是否继续Phase 3

---

## 8. 参考文献与资源

### 8.1 关键文献

**遮挡鲁棒性**:
1. He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
2. Bao et al., "BEiT: BERT Pre-Training of Image Transformers", ICCV 2022
3. Xie et al., "SimMIM: A Simple Framework for Masked Image Modeling", CVPR 2022

**小数据集训练**:
4. Touvron et al., "Training data-efficient image transformers", ICML 2021
5. Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
6. Steiner et al., "How to train your ViT?", arXiv 2021

**CNN vs Transformer**:
7. Liu et al., "ConvNet vs Transformer: A Comparative Study", arXiv 2022
8. Ge et al., "ConvNeXt: A ConvNet for the 2020s", CVPR 2022

### 8.2 代码资源

```
timm库: https://github.com/huggingface/pytorch-image-models
MAE预训练权重: https://github.com/facebookresearch/mae
数据增强: https://github.com/albumentations-team/albumentations
```

---

**文档版本**: 2.0
**状态**: Phase 1 完成,Phase 2 规划中
**下次更新**: Phase 2 Week 1 结束后
