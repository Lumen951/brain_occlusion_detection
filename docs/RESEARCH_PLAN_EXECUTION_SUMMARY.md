# 研究计划执行摘要

**日期**: 2026-01-08
**状态**: Phase 1 完成,Phase 2 规划完成

---

## 📊 当前状态

### Phase 1 实验结果

| 模型 | 平均准确率 | 状态 |
|------|------------|------|
| ViT-B/16 | 52.08% | ❌ 远低于预期(85%+) |
| ResNet-50 | 45.83% | ❌ 远低于预期(70%+) |
| **目标** | **>70%** | ⚠️ 需要改进 |

### 核心问题

1. **数据集太小**: ~5000-7500训练样本,不足以训练深度模型
2. **训练策略不当**: freeze_backbone=True限制了模型学习能力
3. **预训练特征不适配**: ImageNet特征不适合遮挡识别任务

---

## 🎯 Phase 2 计划 (2-3周)

### 目标

将模型性能从 ~50% 提升到 **>70%**

### 三大策略

#### 1. MAE-ViT (最高优先级)

**理由**: 文献强烈支持MAE在遮挡场景的优势
- Masked modeling预训练天然适合遮挡
- 预期提升5-10%准确率

**行动**:
```bash
# 下载MAE权重
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# 训练
python scripts/training/train_model.py --config configs/mae_vit_base.yaml
```

#### 2. 强数据增强

**理由**: 文献证明对小数据集最有效
- Mixup, CutMix, Random Erasing
- 预期提升5-10%

**配置**:
```yaml
augmentation:
  mixup_alpha: 0.2
  cutmix_alpha: 1.0
  random_erasing: 0.3
```

#### 3. 渐进式解冻

**理由**: 比直接微调更适合小数据集
- 4阶段逐步解冻
- 预期提升3-5%

---

## 📚 文献调研关键发现

### 必读论文

1. **He et al., "Masked Autoencoders", CVPR 2022** (3000+引用)
   - MAE通过masked modeling预训练,天然适合遮挡

2. **Steiner et al., "How to train your ViT?", 2021** (800+引用)
   - 小数据集需要强数据增强和渐进式解冻

3. **Geirhos et al., "Texture vs Shape", ICLR 2019** (2000+引用)
   - CNN过度依赖纹理,人类依赖形状

### 核心洞察

- **MAE是最有希望的模型**: 文献和理论都支持
- **小数据集需要特殊策略**: 不能直接套用大数据集方法
- **ViT需要更多数据**: 当前数据集可能不足以发挥ViT优势

---

## 🗓️ 时间线

### Week 1 (本周)

**Day 1-2**: 数据诊断
- 检查类别/遮挡分布
- 识别潜在问题

**Day 3-5**: 三个实验
- ViT全模型微调
- ViT强数据增强
- ResNet全模型微调

**Day 6-7**: 结果对比
- 评估哪个策略最有效

### Week 2

**Day 1-3**: MAE-ViT训练
- 下载MAE权重
- Fine-tune on OIID

**Day 4-5**: 集成学习
- 集成最佳的3个模型
- 测试软投票/硬投票

**Day 6-7**: 错误分析
- Attention可视化
- 识别困难样本

### Week 3

**决策点**: 评估是否达到>70%

- ✅ **>70%** → 进入Phase 3 (5-7个模型对比)
- ⚠️ **60-70%** → 调整Phase 3 (3-4个模型)
- ❌ **<60%** → 转向方法论研究

---

## 🎓 Phase 3 规划 (条件执行)

### 前提

Phase 2达到>70%性能

### 模型选择 (优先级排序)

1. **MAE-ViT** (Phase 2已训练)
2. **ViT-B/16** (重新训练,强增强)
3. **ResNet-50** (重新训练,强增强)
4. **ConvNeXt-B** (现代CNN)
5. **Swin-B** (层级Transformer)

### 分析维度

- 性能对比 (各遮挡等级)
- 错误模式分析 (Jaccard相似度)
- Attention分析 (entropy, focus ratio)
- 表征分析 (RSA)

### 投稿目标

- **最佳**: AAAI 2026 / IJCAI 2026 (5个模型,>70%)
- **中等**: WACV 2026 / BMVC 2026 (3-4个模型,60-70%)
- **保底**: Workshop / ICPR 2026 (方法论研究)

---

## 📋 检查清单

### 本周必做

- [ ] 运行数据诊断脚本
- [ ] 创建4个配置文件 (unfreeze, strong_aug, resnet_unfreeze, mae)
- [ ] 启动3个训练实验
- [ ] 下载MAE预训练权重
- [ ] 修改pretrained_loader.py添加MAE支持

### 下周必做

- [ ] 训练MAE-ViT
- [ ] 测试集成学习
- [ ] Attention可视化
- [ ] 错误样本分析

### Week 3决策

- [ ] 评估所有实验结果
- [ ] 决策: 进入Phase 3 或调整方向
- [ ] 与导师沟通下一步计划

---

## 📁 已创建文档

1. **PHASE1_ANALYSIS_AND_NEXT_STEPS.md** (详细版)
   - 完整的Phase 1分析
   - Phase 2-4详细规划
   - 风险评估与应对

2. **IMMEDIATE_ACTION_PLAN.md** (执行版)
   - Week 1-3具体任务
   - 所有配置文件和脚本代码
   - 决策树和检查清单

3. **LITERATURE_REVIEW_SUMMARY.md** (文献版)
   - 核心文献发现
   - 方法论建议
   - 关键参考文献

4. **RESEARCH_PLAN_EXECUTION_SUMMARY.md** (本文档)
   - 快速概览
   - 核心行动
   - 时间线

---

## 🚀 立即开始

### 第一步: 数据诊断

```bash
python scripts/analysis/diagnose_dataset.py
```

### 第二步: 创建配置文件

复制并修改现有配置:
```bash
cp configs/vit_b16_image_split.yaml configs/vit_b16_unfreeze.yaml
cp configs/vit_b16_image_split.yaml configs/vit_b16_strong_aug.yaml
cp configs/resnet50_image_split.yaml configs/resnet50_unfreeze.yaml
```

### 第三步: 启动训练

```bash
# 实验1: ViT全模型微调
python scripts/training/train_model.py --config configs/vit_b16_unfreeze.yaml

# 实验2: ViT强数据增强
python scripts/training/train_model.py --config configs/vit_b16_strong_aug.yaml

# 实验3: ResNet全模型微调
python scripts/training/train_model.py --config configs/resnet50_unfreeze.yaml
```

---

## 💡 关键建议

### 务实调整

- ❌ 不要继续原计划的10个模型对比
- ✅ 先解决基础性能问题
- ✅ 分阶段产出,降低风险

### 聚焦MAE

- MAE是最有希望的模型
- 文献和理论都强烈支持
- 应该作为Phase 2的核心

### 保持灵活

- 每周都有决策点
- 根据结果及时调整
- 准备多个备选方案

---

## 📞 需要帮助?

如果遇到问题,参考:
- **技术问题**: IMMEDIATE_ACTION_PLAN.md (有完整代码)
- **战略问题**: PHASE1_ANALYSIS_AND_NEXT_STEPS.md (有详细分析)
- **文献问题**: LITERATURE_REVIEW_SUMMARY.md (有文献支持)

---

**创建日期**: 2026-01-08
**预计完成**: 2026-01-29 (3周)
**下次更新**: Week 1结束后 (2026-01-15)
